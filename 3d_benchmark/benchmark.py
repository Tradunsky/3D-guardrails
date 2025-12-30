import time
import os
import io
import sys
import multiprocessing
import numpy as np

# Set EGL platform for headless rendering.
os.environ["PYOPENGL_PLATFORM"] = "egl"

try:
    import pyvista as pv
except ImportError:
    pv = None
try:
    import pygfx as gfx
except ImportError:
    gfx = None
import trimesh
try:
    import pyrender
except ImportError:
    pyrender = None
from PIL import Image
import logging
try:
    from rendercanvas.offscreen import RenderCanvas
except ImportError:
    RenderCanvas = None
try:
    import open3d as o3d
except ImportError:
    o3d = None
import tempfile
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import csv


logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

RES = (512, 512)

# ASSET_PATH = os.path.join(os.path.dirname(__file__), "candy.glb")
TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tests/data"))

VIEW_ANGLES = ((0, 10), (90, 10), (180, 10), (270, 10), (45, 45), (225, 45))
BG_COLOR = [0.05, 0.05, 0.05, 1.0]


def _to_radians(angles):
    vals = list(angles)
    az_deg, el_deg = vals[0], vals[1]
    return (float(np.deg2rad(az_deg)), float(np.deg2rad(el_deg)), 0.0)

def _spherical_to_cartesian(distance, azimuth, elevation):
    x = distance * np.cos(elevation) * np.cos(azimuth)
    z = distance * np.cos(elevation) * np.sin(azimuth)
    y = distance * np.sin(elevation)
    return np.array([x, y, z])

def get_mesh_stats(loaded):
    bounds = loaded.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = np.linalg.norm(extent) / 2.0
    return center, radius

def get_camera_positions(center, radius, distance_multiplier=1.2):
    fov = np.pi / 3.0
    render_distance = (radius / np.sin(fov / 2.0)) * distance_multiplier
    positions = []
    for az_deg, el_deg in VIEW_ANGLES:
        az_rad, el_rad, _ = _to_radians((az_deg, el_deg))
        camera_pos = _spherical_to_cartesian(render_distance, az_rad, el_rad) + center
        positions.append(camera_pos)
    return positions

def get_texture_image(material):
    if material is None: return None
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        return material.baseColorTexture
    if hasattr(material, 'image') and material.image is not None:
        return material.image
    return None

def bench_pyrender(contents):
    if pyrender is None:
        print("❌ Pyrender not installed")
        return 0
    print("🚀 Benchmarking Pyrender (Inc. Loading)...")
    start_total = time.perf_counter()
    file_obj = io.BytesIO(contents)
    loaded = trimesh.load(file_obj, file_type='glb', skip_materials=False)
    center, radius = get_mesh_stats(loaded)
    if isinstance(loaded, trimesh.Scene):
        meshes = [pyrender.Mesh.from_trimesh(g, smooth=False) for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    else: meshes = [pyrender.Mesh.from_trimesh(loaded, smooth=False)]
    scene = pyrender.Scene(bg_color=BG_COLOR, ambient_light=[0.3, 0.3, 0.3])
    for m in meshes: scene.add(m)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    scene.add_node(pyrender.Node(light=pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=6.0), matrix=np.eye(4)))
    scene.add(pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=3.5), pose=trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    scene.add(pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5), pose=trimesh.transformations.rotation_matrix(np.pi/3, [0, 1, 0]))
    r = pyrender.OffscreenRenderer(RES[0], RES[1])
    cam_positions = get_camera_positions(center, radius)
    try:
        total_img = Image.new('RGB', (RES[0] * 3, RES[1] * 2))
        for idx, pos in enumerate(cam_positions):
            f = (center - pos) / np.linalg.norm(center - pos)
            u, s = np.array([0, 1, 0]), np.cross(f, np.array([0, 1, 0]))
            if np.linalg.norm(s) < 1e-3: s = np.cross(f, np.array([1, 0, 0]))
            s /= np.linalg.norm(s)
            u = np.cross(s, f)
            pose = np.eye(4)
            pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = s, u, -f, pos
            scene.set_pose(camera_node, pose)
            color, _ = r.render(scene)
            view_img = Image.fromarray(color)
            row, col = divmod(idx, 3)
            total_img.paste(view_img, (col * RES[0], row * RES[1]))            
        
        with io.BytesIO() as output:
            total_img.save(output, format="PNG")
            png_bytes = output.getvalue()
            if os.getenv("DEBUG"): total_img.save(f"pyrender_tiled.png")
    finally: r.delete()
    return (time.perf_counter() - start_total) * 1000, png_bytes

def bench_pyvista(contents):
    if pv is None:
        print("❌ PyVista not installed")
        return 0
    print("🚀 Benchmarking PyVista (Inc. Loading)...")    
    start_total = time.perf_counter()
    file_obj = io.BytesIO(contents)
    loaded = trimesh.load(file_obj, file_type='glb', skip_materials=False)
    center, radius = get_mesh_stats(loaded)
    cam_positions = get_camera_positions(center, radius)
    pl = pv.Plotter(off_screen=True, window_size=RES, lighting=None)
    if isinstance(loaded, trimesh.Scene):
        for name, g in loaded.geometry.items():
            if isinstance(g, trimesh.Trimesh): 
                mesh = pv.wrap(g)
                tex = None
                if hasattr(g.visual, 'material'):
                    image = get_texture_image(g.visual.material)
                    if image is not None:
                        tex = pv.Texture(np.array(image))
                pl.add_mesh(mesh, texture=tex)
    else: 
        mesh = pv.wrap(loaded)
        tex = None
        if hasattr(loaded.visual, 'material'):
            image = get_texture_image(loaded.visual.material)
            if image is not None:
                tex = pv.Texture(np.array(image))
        pl.add_mesh(mesh, texture=tex)
    pl.background_color = BG_COLOR[:3]
    pl.add_light(pv.Light(position=(0, 0, 1), color='white', intensity=1.5, light_type='camera light'))
    pl.add_light(pv.Light(position=(0, 1, 0), color=[0.9, 0.95, 1.0], intensity=1.0))
    pl.add_light(pv.Light(position=(1, 0, 0), color=[1.0, 0.95, 0.9], intensity=0.7))
    views = []
    for idx, pos in enumerate(cam_positions):
        pl.camera_position = [pos, center, (0.0, 1.0, 0.0)]
        pl.camera.view_angle = 60
        pl.render()
        img = pl.screenshot(None, return_img=True)
        view_img = Image.fromarray(img)
        views.append(view_img)        
    
    # Stitch them: 2 rows, 3 columns to match rendering.py
    tiled_img = Image.new('RGB', (RES[0] * 3, RES[1] * 2), color=(0, 0, 0))
    for i, img in enumerate(views):
        row, col = divmod(i, 3)
        tiled_img.paste(img, (col * RES[0], row * RES[1]))

    with io.BytesIO() as output:
        tiled_img.save(output, format="PNG")
        png_bytes = output.getvalue()
        if os.getenv("DEBUG"): tiled_img.save(f"pyvista_tiled.png")
    pl.close()
    return (time.perf_counter() - start_total) * 1000, png_bytes

def bench_pygfx(contents):
    if gfx is None or RenderCanvas is None:
        print("❌ PyGfx or RenderCanvas not installed")
        return 0
    print("🚀 Benchmarking PyGfx (Inc. Loading)...")
    start_total = time.perf_counter()
    file_obj = io.BytesIO(contents)
    loaded = trimesh.load(file_obj, file_type='glb', skip_materials=False)
    center, radius = get_mesh_stats(loaded)
    try:
        canvas = RenderCanvas(size=RES, pixel_ratio=1)
        renderer = gfx.renderers.WgpuRenderer(canvas)
    except Exception as e:
        print(f"Skipping PyGfx due to init error: {e}")
        return 0
    scene = gfx.Scene()
    if isinstance(loaded, trimesh.Scene):
        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                geo = gfx.geometry_from_trimesh(g)
                mat = gfx.MeshStandardMaterial()
                image = get_texture_image(g.visual.material) if hasattr(g.visual, 'material') else None
                if image is not None:
                    mat.map = gfx.Texture(np.array(image), dim=2)
                scene.add(gfx.Mesh(geo, mat))
    else: 
        geo = gfx.geometry_from_trimesh(loaded)
        mat = gfx.MeshStandardMaterial()
        image = get_texture_image(loaded.visual.material) if hasattr(loaded.visual, 'material') else None
        if image is not None:
            mat.map = gfx.Texture(np.array(image), dim=2)
        scene.add(gfx.Mesh(geo, mat))
    scene.add(gfx.AmbientLight(intensity=0.3))
    key_light = gfx.DirectionalLight(intensity=1.5)
    key_light.local.position = (0, 0, 1)
    scene.add(key_light)
    camera = gfx.PerspectiveCamera(60, 1)
    cam_positions = get_camera_positions(center, radius)
    total_img = Image.new('RGB', (RES[0] * 3, RES[1] * 2))
    for idx, pos in enumerate(cam_positions):
        camera.local.position = pos
        camera.look_at(center)
        renderer.render(scene, camera)
        canvas.draw()
        img = renderer.snapshot()
        view_img = Image.fromarray(img)
        row, col = divmod(idx, 3)
        total_img.paste(view_img, (col * RES[0], row * RES[1]))
        
    with io.BytesIO() as output:
        total_img.save(output, format="PNG")
        png_bytes = output.getvalue()
        if os.getenv("DEBUG"): total_img.save(f"pygfx_tiled.png")
    return (time.perf_counter() - start_total) * 1000, png_bytes

def bench_open3d(contents):
    if o3d is None:
        print("❌ Open3D not installed")
        return 0
    print("🚀 Benchmarking Open3D (Inc. Loading)...")
    start_total = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tf:
        tf.write(contents)
        tmp_path = tf.name
    try:
        model = o3d.io.read_triangle_model(tmp_path)
        render = o3d.visualization.rendering.OffscreenRenderer(RES[0], RES[1])
        render.scene.set_background(BG_COLOR)
        
        # Add the whole model at once - this preserves materials from GLB
        render.scene.add_model("model", model)
        
        # Lighting (Filament uses Lux, sun is ~100k)
        render.scene.scene.set_sun_light([0.577, 0.577, 0.577], [1, 1, 1], 100000)
        render.scene.scene.enable_sun_light(True)
        
        # Calculate bounding box from all meshes
        full_bbox = o3d.geometry.AxisAlignedBoundingBox()
        for mesh_node in model.meshes:
            full_bbox += mesh_node.mesh.get_axis_aligned_bounding_box()

        center = full_bbox.get_center()
        radius = np.linalg.norm(full_bbox.get_max_bound() - full_bbox.get_min_bound()) / 2.0
        cam_positions = get_camera_positions(center, radius)
        total_img = Image.new('RGB', (RES[0] * 3, RES[1] * 2))
        for idx, pos in enumerate(cam_positions):
            render.setup_camera(60.0, center, pos, [0, 1, 0])
            img = render.render_to_image()
            img_np = np.asarray(img)
            view_img = Image.fromarray(img_np)
            row, col = divmod(idx, 3)
            total_img.paste(view_img, (col * RES[0], row * RES[1]))

        with io.BytesIO() as output:
            total_img.save(output, format="PNG")
            png_bytes = output.getvalue()
            if os.getenv('DEBUG'): total_img.save(f'open3d_tiled.png')
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
    return (time.perf_counter() - start_total) * 1000, png_bytes

def run_bench(name, func, contents, results):
    try:
        res = func(contents)
        # res is either (latency, bytes) or (latency, None)
        # We need to return it as is or slightly wrapped
        results[name] = res
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ {name} failed: {e}")


def generate_charts(all_latencies):
    # all_latencies: {framework: [all_raw_latencies]}
    frameworks = ['Pyrender', 'PyGfx', 'PyVista', 'Open3D'] # Keep order
    
    means = []
    mins = []
    maxs = []
    labels = []

    print("\n--- Aggregate Statistics ---")
    for fw in frameworks:
        data = all_latencies.get(fw, [])
        # Filter out failed runs
        valid_data = [d for d in data if isinstance(d, (int, float)) and d > 0]
        
        if valid_data:
            avg = np.mean(valid_data)
            minimum = np.min(valid_data)
            maximum = np.max(valid_data)
            
            means.append(avg)
            # Error bars: [mean - min, max - mean]
            mins.append(avg - minimum) 
            maxs.append(maximum - avg)
            labels.append(fw)
            print(f"{fw}: Mean={avg:.2f}ms, Min={minimum:.2f}ms, Max={maximum:.2f}ms")
        else:
            print(f"{fw}: No valid data")
            means.append(0)
            mins.append(0)
            maxs.append(0)
            labels.append(fw)

    x = np.arange(len(frameworks))
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    # Error bars format: [bottom_errors, top_errors]
    yerr = [mins, maxs]
    
    bars = ax.bar(x, means, yerr=yerr, align='center', alpha=0.8, ecolor='black', capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.bar_label(bars, fmt='%.1f', padding=5)
    
    ax.set_ylabel('Total Latency (6 views) (ms)')
    ax.set_title('Total 3D Rendering Latency for 6 Views (Lower is better)\nError bars show Min/Max range')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.png')
    plt.savefig(output_path)
    print(f"📊 Aggregate Chart saved to {output_path}")
    plt.close()



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='3D Benchmark')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations per mesh')
    args = parser.parse_args()
    
    # 1. Discover GLB files
    glb_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.glb"))
    if not glb_files:
        print(f"No .glb files found in {TEST_DATA_DIR}")
        sys.exit(1)
        
    print(f"Found {len(glb_files)} GLB files. Running {args.iterations} iterations each.")
    
    # Structure: [Mesh, Framework, Iteration, Latency]
    csv_data = []
    
    # Structure: {mesh_name: {framework: median_latency}}
    # Structure: {framework: [all_latencies]}
    all_latencies = {}  

    # Use a single Manager context for the whole run to avoid process leaks
    with multiprocessing.Manager() as manager:
        for glb_path in glb_files:
            mesh_name = os.path.basename(glb_path)
            print(f"\n--- Benchmarking {mesh_name} ---")
            
            with open(glb_path, "rb") as f:
                glb_contents = f.read()

            mesh_latencies = {} # {framework: [latencies]}

            for i in range(args.iterations):
                # print(f"  Iteration {i+1}/{args.iterations}")
                results = manager.dict()
                all_benches = [
                    ('Pyrender', bench_pyrender), 
                    ('PyGfx', bench_pygfx), 
                    ('PyVista', bench_pyvista),
                    ('Open3D', bench_open3d),
                ]
                
                for name, func in all_benches:
                    p = multiprocessing.Process(target=run_bench, args=(name, func, glb_contents, results))
                    p.start()
                    p.join()
                
                for name, _ in all_benches:
                    res_tuple = results.get(name, 0)
                    latency = 0
                    if isinstance(res_tuple, tuple):
                        latency, _ = res_tuple
                    elif isinstance(res_tuple, (int, float)):
                        latency = res_tuple
                    
                    if name not in all_latencies: all_latencies[name] = []

                    if latency > 0:
                        csv_data.append([mesh_name, name, i+1, latency])
                        if name not in mesh_latencies: mesh_latencies[name] = []
                        mesh_latencies[name].append(latency)
                        all_latencies[name].append(latency)
                    else:
                        csv_data.append([mesh_name, name, i+1, "FAILED"])
                        all_latencies[name].append("FAILED")

            # Compute stats for this mesh
            print(f'Results for {mesh_name} (Median of {args.iterations} runs):')
            for name in ['Pyrender', 'PyGfx', 'PyVista', 'Open3D']:
                lats = mesh_latencies.get(name, [])
                if lats:
                    median_lat = np.median(lats)
                    print(f'{name}: {median_lat:.2f} ms')
                else:
                    print(f'{name}: FAILED')
        

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mesh', 'Framework', 'Iteration', 'Latency_ms'])
        writer.writerows(csv_data)
    print(f"\n📝 Full results saved to {csv_path}")

    print('\n------------------------------------------------------------------')
    generate_charts(all_latencies)
    print('------------------------------------------------------------------')

