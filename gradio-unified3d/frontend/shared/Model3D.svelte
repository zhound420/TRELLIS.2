<script lang="ts">
	import type { FileData } from "@gradio/client";
	import { BlockLabel, IconButton, IconButtonWrapper } from "@gradio/atoms";
	import { File, Download, Undo } from "@gradio/icons";
	import type { I18nFormatter } from "@gradio/utils";
	import { dequal } from "dequal";
	import Canvas3D from "./Canvas3D.svelte";

	export let value: FileData | null;
	export let display_mode: "solid" | "point_cloud" | "wireframe" = "solid";
	export let clear_color: [number, number, number, number] = [0, 0, 0, 0];
	export let label = "";
	export let show_label: boolean;
	export let i18n: I18nFormatter;
	export let zoom_speed = 1;
	export let pan_speed = 1;
	export let camera_position: [number | null, number | null, number | null] = [
		null,
		null,
		null
	];
	export let has_change_history = false;
	export let hdri_url: string | null = null;
	export let scale_mm: number = 100;
	export let show_measurements: boolean = false;

	let current_settings = { camera_position, zoom_speed, pan_speed };

	let canvas3d: Canvas3D | undefined;
	function handle_undo(): void {
		canvas3d?.reset_camera_position();
	}

	$: {
		if (
			!dequal(current_settings.camera_position, camera_position) ||
			current_settings.zoom_speed !== zoom_speed ||
			current_settings.pan_speed !== pan_speed
		) {
			canvas3d?.update_camera(camera_position, zoom_speed, pan_speed);
			current_settings = { camera_position, zoom_speed, pan_speed };
		}
	}
</script>

<BlockLabel
	{show_label}
	Icon={File}
	label={label || i18n("3D_model.3d_model")}
/>
{#if value}
	<div class="model3D" data-testid="model3d">
		<IconButtonWrapper>
			<IconButton
				Icon={Undo}
				label="Undo"
				on:click={() => handle_undo()}
				disabled={!has_change_history}
			/>
			<a
				href={value.url}
				target={window.__is_colab__ ? "_blank" : null}
				download={window.__is_colab__ ? null : value.orig_name || value.path}
			>
				<IconButton Icon={Download} label={i18n("common.download")} />
			</a>
		</IconButtonWrapper>

		<Canvas3D
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
	.model3D {
		display: flex;
		position: relative;
		width: var(--size-full);
		height: var(--size-full);
		border-radius: var(--block-radius);
		overflow: hidden;
	}
	.model3D :global(canvas) {
		width: var(--size-full);
		height: var(--size-full);
		object-fit: contain;
		overflow: hidden;
	}
</style>
