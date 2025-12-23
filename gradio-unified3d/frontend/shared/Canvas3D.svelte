<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import type { FileData } from "@gradio/client";
	import * as THREE from "three";
	import { OrbitControls } from "three/addons/controls/OrbitControls.js";
	import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
	import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
	import { ThreeMFLoader } from "three/addons/loaders/3MFLoader.js";
	import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
	import { EXRLoader } from "three/addons/loaders/EXRLoader.js";

	export let value: FileData;
	export let display_mode: "solid" | "point_cloud" | "wireframe" = "solid";
	export let clear_color: [number, number, number, number] = [0.1, 0.1, 0.1, 1];
	export let camera_position: [number | null, number | null, number | null] = [null, null, null];
	export let zoom_speed: number = 1;
	export let pan_speed: number = 1;
	export let hdri_url: string | null = null;
	export let scale_mm: number = 100;
	export let show_measurements: boolean = false;

	$: url = value?.url;

	let container: HTMLDivElement;
	let scene: THREE.Scene;
	let camera: THREE.PerspectiveCamera;
	let renderer: THREE.WebGLRenderer;
	let controls: OrbitControls;
	let currentModel: THREE.Group | null = null;
	let animationId: number;
	let mounted = false;

	// Loaders
	let gltfLoader: GLTFLoader;
	let threemfLoader: ThreeMFLoader;
	let dracoLoader: DRACOLoader;

	function initScene(): void {
		// Scene
		scene = new THREE.Scene();
		scene.background = new THREE.Color(clear_color[0], clear_color[1], clear_color[2]);

		// Camera
		const aspect = container.clientWidth / container.clientHeight;
		camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 1000);
		camera.position.set(2, 2, 2);

		// Renderer
		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setSize(container.clientWidth, container.clientHeight);
		renderer.setPixelRatio(window.devicePixelRatio);
		renderer.toneMapping = THREE.ACESFilmicToneMapping;
		renderer.toneMappingExposure = 1;
		renderer.outputColorSpace = THREE.SRGBColorSpace;
		container.appendChild(renderer.domElement);

		// Controls
		controls = new OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.05;
		controls.zoomSpeed = zoom_speed;
		controls.panSpeed = pan_speed;
		controls.minDistance = 0.1;
		controls.maxDistance = 100;

		// Lighting (fallback if no HDRI)
		const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
		scene.add(ambientLight);

		const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
		directionalLight.position.set(5, 10, 7.5);
		scene.add(directionalLight);

		// Initialize loaders
		gltfLoader = new GLTFLoader();
		dracoLoader = new DRACOLoader();
		dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
		gltfLoader.setDRACOLoader(dracoLoader);

		threemfLoader = new ThreeMFLoader();

		// Load HDRI if provided
		if (hdri_url) {
			loadEnvironment(hdri_url);
		}

		// Handle resize
		window.addEventListener('resize', onWindowResize);
	}

	async function loadEnvironment(envUrl: string): Promise<void> {
		try {
			let texture: THREE.Texture;

			if (envUrl.endsWith('.exr')) {
				const exrLoader = new EXRLoader();
				texture = await exrLoader.loadAsync(envUrl);
			} else {
				const rgbeLoader = new RGBELoader();
				texture = await rgbeLoader.loadAsync(envUrl);
			}

			texture.mapping = THREE.EquirectangularReflectionMapping;
			scene.environment = texture;
			// Optionally use as background
			// scene.background = texture;
		} catch (error) {
			console.warn('Failed to load HDRI environment:', error);
		}
	}

	async function loadModel(modelUrl: string): Promise<void> {
		if (!modelUrl) return;

		// Remove existing model
		if (currentModel) {
			scene.remove(currentModel);
			currentModel.traverse((child) => {
				if (child instanceof THREE.Mesh) {
					child.geometry?.dispose();
					if (Array.isArray(child.material)) {
						child.material.forEach(m => m.dispose());
					} else {
						child.material?.dispose();
					}
				}
			});
			currentModel = null;
		}

		try {
			const ext = modelUrl.split('.').pop()?.toLowerCase() || '';
			let model: THREE.Group;

			if (ext === 'glb' || ext === 'gltf') {
				const gltf = await gltfLoader.loadAsync(modelUrl);
				model = gltf.scene;
			} else if (ext === '3mf') {
				model = await threemfLoader.loadAsync(modelUrl);
			} else {
				// Try GLTF loader as default
				const gltf = await gltfLoader.loadAsync(modelUrl);
				model = gltf.scene;
			}

			// Apply display mode
			if (display_mode === 'wireframe') {
				model.traverse((child) => {
					if (child instanceof THREE.Mesh) {
						child.material = new THREE.MeshBasicMaterial({
							color: 0x00ff00,
							wireframe: true
						});
					}
				});
			} else if (display_mode === 'point_cloud') {
				model.traverse((child) => {
					if (child instanceof THREE.Mesh) {
						const points = new THREE.Points(
							child.geometry,
							new THREE.PointsMaterial({ color: 0x00ff00, size: 0.01 })
						);
						child.parent?.add(points);
						child.visible = false;
					}
				});
			}

			// Center and scale model
			const box = new THREE.Box3().setFromObject(model);
			const center = box.getCenter(new THREE.Vector3());
			const size = box.getSize(new THREE.Vector3());
			const maxDim = Math.max(size.x, size.y, size.z);

			model.position.sub(center);

			// Fit camera to model
			const fitOffset = 1.2;
			const fitHeightDistance = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360));
			const fitWidthDistance = fitHeightDistance / camera.aspect;
			const distance = fitOffset * Math.max(fitHeightDistance, fitWidthDistance);

			camera.position.set(distance, distance * 0.5, distance);
			controls.target.set(0, 0, 0);
			controls.update();

			scene.add(model);
			currentModel = model;
		} catch (error) {
			console.error('Failed to load model:', error);
		}
	}

	function onWindowResize(): void {
		if (!container || !camera || !renderer) return;

		camera.aspect = container.clientWidth / container.clientHeight;
		camera.updateProjectionMatrix();
		renderer.setSize(container.clientWidth, container.clientHeight);
	}

	function animate(): void {
		animationId = requestAnimationFrame(animate);
		controls?.update();
		renderer?.render(scene, camera);
	}

	export function update_camera(
		cam_position: [number | null, number | null, number | null],
		z_speed: number,
		p_speed: number
	): void {
		if (!controls || !camera) return;

		controls.zoomSpeed = z_speed;
		controls.panSpeed = p_speed;

		// Convert spherical coordinates if provided
		if (cam_position[0] !== null || cam_position[1] !== null || cam_position[2] !== null) {
			const radius = cam_position[2] ?? camera.position.length();
			const alpha = cam_position[0] !== null ? (cam_position[0] * Math.PI) / 180 : Math.atan2(camera.position.z, camera.position.x);
			const beta = cam_position[1] !== null ? (cam_position[1] * Math.PI) / 180 : Math.acos(camera.position.y / radius);

			camera.position.x = radius * Math.sin(beta) * Math.cos(alpha);
			camera.position.y = radius * Math.cos(beta);
			camera.position.z = radius * Math.sin(beta) * Math.sin(alpha);
			controls.update();
		}
	}

	export function reset_camera_position(): void {
		if (currentModel) {
			const box = new THREE.Box3().setFromObject(currentModel);
			const size = box.getSize(new THREE.Vector3());
			const maxDim = Math.max(size.x, size.y, size.z);
			const distance = maxDim * 2;

			camera.position.set(distance, distance * 0.5, distance);
			controls.target.set(0, 0, 0);
			controls.update();
		}
	}

	onMount(() => {
		initScene();
		animate();
		mounted = true;
	});

	onDestroy(() => {
		window.removeEventListener('resize', onWindowResize);
		cancelAnimationFrame(animationId);

		controls?.dispose();
		renderer?.dispose();

		if (currentModel) {
			currentModel.traverse((child) => {
				if (child instanceof THREE.Mesh) {
					child.geometry?.dispose();
					if (Array.isArray(child.material)) {
						child.material.forEach(m => m.dispose());
					} else {
						child.material?.dispose();
					}
				}
			});
		}
	});

	$: if (mounted && url) {
		loadModel(url);
	}

	$: if (mounted && controls) {
		controls.zoomSpeed = zoom_speed;
		controls.panSpeed = pan_speed;
	}
</script>

<div bind:this={container} class="canvas-container"></div>

<style>
	.canvas-container {
		width: 100%;
		height: 100%;
		min-height: 400px;
	}
</style>
