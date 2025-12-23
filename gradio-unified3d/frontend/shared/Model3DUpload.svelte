<script lang="ts">
	import { createEventDispatcher, tick } from "svelte";
	import { Upload, ModifyUpload } from "@gradio/upload";
	import type { FileData, Client } from "@gradio/client";
	import { BlockLabel } from "@gradio/atoms";
	import { File } from "@gradio/icons";
	import type { I18nFormatter } from "@gradio/utils";
	import type Canvas3D from "./Canvas3D.svelte";

	export let value: null | FileData;
	export let display_mode: "solid" | "point_cloud" | "wireframe" = "solid";
	export let clear_color: [number, number, number, number] = [0, 0, 0, 0];
	export let label = "";
	export let show_label: boolean;
	export let root: string;
	export let i18n: I18nFormatter;
	export let zoom_speed = 1;
	export let pan_speed = 1;
	export let max_file_size: number | null = null;
	export let uploading = false;
	export let upload_promise: Promise<(FileData | null)[]> | null = null;
	export let hdri_url: string | null = null;
	export let scale_mm: number = 100;
	export let show_measurements: boolean = false;

	// alpha, beta, radius
	export let camera_position: [number | null, number | null, number | null] = [
		null,
		null,
		null
	];
	export let upload: Client["upload"];
	export let stream_handler: Client["stream"];

	async function handle_upload({
		detail
	}: CustomEvent<FileData>): Promise<void> {
		value = detail;
		await tick();
		dispatch("change", value);
		dispatch("load", value);
	}

	async function handle_clear(): Promise<void> {
		value = null;
		await tick();
		dispatch("clear");
		dispatch("change");
	}

	let Canvas3DComponent: typeof Canvas3D;
	async function loadCanvas3D(): Promise<typeof Canvas3D> {
		const module = await import("./Canvas3D.svelte");
		return module.default;
	}
	$: if (value) {
		loadCanvas3D().then((component) => {
			Canvas3DComponent = component;
		});
	}

	let canvas3d: Canvas3D | undefined;
	async function handle_undo(): Promise<void> {
		canvas3d?.reset_camera_position();
	}

	const dispatch = createEventDispatcher<{
		change: FileData | null;
		clear: undefined;
		drag: boolean;
		load: FileData;
	}>();

	let dragging = false;

	$: dispatch("drag", dragging);
</script>

<BlockLabel {show_label} Icon={File} label={label || "3D Model"} />

{#if value == null}
	<Upload
		bind:upload_promise
		{upload}
		{stream_handler}
		on:load={handle_upload}
		{root}
		{max_file_size}
		filetype={[".stl", ".obj", ".gltf", ".glb", ".3mf", "model/obj"]}
		bind:dragging
		bind:uploading
		on:error
		aria_label={i18n("model3d.drop_to_upload")}
	>
		<slot />
	</Upload>
{:else}
	<div class="input-model">
		<ModifyUpload
			undoable={true}
			on:clear={handle_clear}
			{i18n}
			on:undo={handle_undo}
		/>

		<svelte:component
			this={Canvas3DComponent}
			bind:this={canvas3d}
			{value}
			{display_mode}
			{clear_color}
			{camera_position}
			{zoom_speed}
			{pan_speed}
			{hdri_url}
			{scale_mm}
			{show_measurements}
		/>
	</div>
{/if}

<style>
	.input-model {
		display: flex;
		position: relative;
		justify-content: center;
		align-items: center;
		width: var(--size-full);
		height: var(--size-full);
		border-radius: var(--block-radius);
		overflow: hidden;
	}

	.input-model :global(canvas) {
		width: var(--size-full);
		height: var(--size-full);
		object-fit: contain;
		overflow: hidden;
	}
</style>
