from typing import *
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh


@dataclass
class ProcessedMeshData:
    """Container for processed mesh data ready for export."""
    vertices: np.ndarray      # (N, 3) vertex positions
    faces: np.ndarray         # (M, 3) face indices
    uvs: np.ndarray           # (N, 2) UV coordinates
    normals: np.ndarray       # (N, 3) vertex normals
    base_color: np.ndarray    # (H, W, 3) base color texture
    metallic: np.ndarray      # (H, W, 1) metallic texture
    roughness: np.ndarray     # (H, W, 1) roughness texture
    alpha: np.ndarray         # (H, W, 1) alpha texture


def _process_mesh_for_export(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
) -> ProcessedMeshData:
    """
    Process mesh for export: cleaning, optional remeshing, UV unwrapping, and texture baking.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar

    Returns:
        ProcessedMeshData containing vertices, faces, uvs, normals, and PBR textures
    """
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3

    if use_tqdm:
        pbar = tqdm(total=6, desc="Processing mesh for export")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()

    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)

    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")

    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 5: Unify face orientations
        mesh.unify_face_orientations()

    # --- Branch 2: Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()

        # Perform Dual Contouring remeshing (rebuilds topology)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Simplify and clean the remeshed result (similar logic to above)
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")


    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)

    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)

    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0

    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # --- Texture Post-Processing ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)

    mask_np = mask.cpu().numpy()

    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask_np).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    # --- Coordinate System Conversion ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()

    # Swap Y and Z axes, invert Y (common conversion for export compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    return ProcessedMeshData(
        vertices=vertices_np,
        faces=faces_np,
        uvs=uvs_np,
        normals=normals_np,
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        alpha=alpha,
    )


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar

    Returns:
        trimesh.Trimesh: A textured mesh ready for export
    """
    # Process mesh using shared helper
    data = _process_mesh_for_export(
        vertices=vertices,
        faces=faces,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
        mesh_cluster_refine_iterations=mesh_cluster_refine_iterations,
        mesh_cluster_global_iterations=mesh_cluster_global_iterations,
        mesh_cluster_smooth_strength=mesh_cluster_smooth_strength,
        verbose=verbose,
        use_tqdm=use_tqdm,
    )

    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([data.base_color, data.alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(data.metallic), data.roughness, data.metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=True if not remesh else False,
    )

    textured_mesh = trimesh.Trimesh(
        vertices=data.vertices,
        faces=data.faces,
        vertex_normals=data.normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=data.uvs, material=material)
    )

    return textured_mesh


def to_3mf(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    output_path: str = None,
    scale_mm: float = 100.0,
    verbose: bool = False,
    use_tqdm: bool = False,
) -> bytes:
    """
    Convert an extracted mesh to a 3MF file with embedded textures for 3D printing.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        output_path: path to write the 3MF file (if None, returns bytes)
        scale_mm: output scale in millimeters (default 100mm = 10cm model)
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar

    Returns:
        bytes: 3MF file data (if output_path is None), otherwise None
    """
    import lib3mf
    import io

    # Process mesh using shared helper
    data = _process_mesh_for_export(
        vertices=vertices,
        faces=faces,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
        mesh_cluster_refine_iterations=mesh_cluster_refine_iterations,
        mesh_cluster_global_iterations=mesh_cluster_global_iterations,
        mesh_cluster_smooth_strength=mesh_cluster_smooth_strength,
        verbose=verbose,
        use_tqdm=use_tqdm,
    )

    if verbose:
        print("Building 3MF file...")

    # Create lib3mf wrapper and model
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()

    # Set model unit to millimeters
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)

    # Create mesh object
    mesh_obj = model.AddMeshObject()
    mesh_obj.SetName("generated_mesh")

    # Scale vertices to mm (input is normalized [-0.5, 0.5])
    # Scale factor converts to desired mm size
    vertices_mm = data.vertices * scale_mm

    # Add vertices
    vertices_list = []
    for v in vertices_mm:
        pos = lib3mf.Position()
        pos.Coordinates[0] = float(v[0])
        pos.Coordinates[1] = float(v[1])
        pos.Coordinates[2] = float(v[2])
        vertices_list.append(pos)
    mesh_obj.SetGeometry(vertices_list, [])

    # Add triangles
    triangles_list = []
    for f in data.faces:
        tri = lib3mf.Triangle()
        tri.Indices[0] = int(f[0])
        tri.Indices[1] = int(f[1])
        tri.Indices[2] = int(f[2])
        triangles_list.append(tri)
    mesh_obj.SetGeometry(vertices_list, triangles_list)

    # Create base color texture as PNG
    base_color_rgba = np.concatenate([data.base_color, data.alpha], axis=-1)
    base_color_img = Image.fromarray(base_color_rgba)

    # Save texture to bytes
    texture_buffer = io.BytesIO()
    base_color_img.save(texture_buffer, format='PNG')
    texture_bytes = texture_buffer.getvalue()

    # Add texture attachment to model with correct 3D Texture relationship type
    texture_path = "/3D/Textures/basecolor.png"
    attachment = model.AddAttachment(texture_path, "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture")
    attachment.ReadFromBuffer(texture_bytes)

    # Create Texture2D resource
    texture_2d = model.AddTexture2DFromAttachment(attachment)
    texture_2d.SetContentType(lib3mf.TextureType.PNG)
    texture_2d.SetTileStyleUV(lib3mf.TextureTileStyle.Wrap, lib3mf.TextureTileStyle.Wrap)

    # Create Texture2DGroup with UV coordinates
    tex_group = model.AddTexture2DGroup(texture_2d)

    # Add UV coordinates to the texture group
    # For each vertex, add its UV coordinate
    uv_indices = []
    for uv in data.uvs:
        tex_coord = lib3mf.Tex2Coord()
        tex_coord.U = float(uv[0])
        tex_coord.V = float(uv[1])
        idx = tex_group.AddTex2Coord(tex_coord)
        uv_indices.append(idx)

    # Set triangle properties (assign texture coordinates to each triangle)
    for i, f in enumerate(data.faces):
        tri_props = lib3mf.TriangleProperties()
        tri_props.ResourceID = tex_group.GetResourceID()
        tri_props.PropertyIDs[0] = uv_indices[int(f[0])]
        tri_props.PropertyIDs[1] = uv_indices[int(f[1])]
        tri_props.PropertyIDs[2] = uv_indices[int(f[2])]
        mesh_obj.SetTriangleProperties(i, tri_props)

    # Set object-level property to use the texture group (using first UV index as default)
    if uv_indices:
        mesh_obj.SetObjectLevelProperty(tex_group.GetResourceID(), uv_indices[0])

    # Add mesh to build items (required for 3MF)
    identity_transform = wrapper.GetIdentityTransform()
    model.AddBuildItem(mesh_obj, identity_transform)

    # Write 3MF file
    writer = model.QueryWriter("3mf")

    if output_path is not None:
        writer.WriteToFile(output_path)
        if verbose:
            print(f"3MF file written to: {output_path}")
        return None
    else:
        # Write to buffer and return bytes
        buffer_size = 10 * 1024 * 1024  # 10MB initial buffer
        output_buffer = writer.WriteToBuffer()
        if verbose:
            print(f"3MF file size: {len(output_buffer)} bytes")
        return bytes(output_buffer)