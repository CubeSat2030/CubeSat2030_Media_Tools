import os
import json
import traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
import trimesh
import imageio

# ------------------------------------------------------------
# Configuration persistence
# ------------------------------------------------------------
CONFIG_FILE = "orbital_anim_config.json"
DEFAULT_CONFIG = {
    "input_path": "",
    "output_path": "",
    "width": 800,
    "height": 600,
    "total_frames": 120,
    "orbit_speed": 360,
    "model_color": [179, 179, 179],
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        # Ensure all expected keys exist
        for key, val in DEFAULT_CONFIG.items():
            if key not in cfg:
                cfg[key] = val
        return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ------------------------------------------------------------
# Model loading (unchanged)
# ------------------------------------------------------------
def load_model(filepath):
    try:
        mesh = trimesh.load(filepath, force='mesh')
    except Exception:
        try:
            import meshio
            m = meshio.read(filepath)
            verts = m.points
            faces = None
            for ct, cd in m.cells_dict.items():
                if ct == "triangle":
                    faces = cd
                    break
            if faces is None:
                raise ValueError("No triangles found.")
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        except ImportError:
            raise RuntimeError("Install meshio for 3MF support: pip install meshio")
        except Exception:
            raise RuntimeError(f"Failed to load model:\n{traceback.format_exc()}")

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No trimesh objects in scene.")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a single mesh.")
    return mesh

# ------------------------------------------------------------
# Software rasterizer (accepts a base color)
# ------------------------------------------------------------
class SoftwareRenderer:
    def __init__(self, width, height, base_color_rgb, fov=np.pi/3,
                 light_dir=np.array([0.5, 0.5, 1.0])):
        self.width = width
        self.height = height
        self.fov = fov
        self.light_dir = light_dir / np.linalg.norm(light_dir)
        self.base_color = np.array(base_color_rgb, dtype=np.uint8)
        aspect = width / height
        self.proj = self._perspective_projection(fov, aspect, near=0.1, far=100.0)

    @staticmethod
    def _perspective_projection(fov, aspect, near, far):
        f = 1.0 / np.tan(fov / 2)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def render(self, vertices, faces, camera_pos, camera_target=np.array([0,0,0])):
        # View matrix
        z_axis = camera_pos - camera_target
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0,1,0]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[:3, 3] = -np.dot(view[:3, :3], camera_pos)

        mvp = self.proj @ view
        verts_h = np.hstack([vertices, np.ones((len(vertices), 1))]).astype(np.float32)
        clip = verts_h @ mvp.T

        w = clip[:, 3:4].copy()
        w[w == 0] = 1e-10
        ndc = clip[:, :3] / w

        screen_x = ((ndc[:, 0] + 1) * 0.5 * self.width).astype(int)
        screen_y = ((1 - ndc[:, 1]) * 0.5 * self.height).astype(int)

        tri_verts = vertices[faces]
        normals = np.cross(tri_verts[:,1]-tri_verts[:,0], tri_verts[:,2]-tri_verts[:,0])
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
        light_intensity = np.dot(normals, self.light_dir)
        light_intensity = np.clip(light_intensity, 0.2, 1.0)

        screen_tris = np.stack([screen_x[faces], screen_y[faces]], axis=-1)
        e1 = screen_tris[:,1] - screen_tris[:,0]
        e2 = screen_tris[:,2] - screen_tris[:,0]
        cross_z = e1[:,0]*e2[:,1] - e1[:,1]*e2[:,0]
        front_facing = cross_z < 0

        visible_faces = faces[front_facing]
        vis_light = light_intensity[front_facing]
        vis_screen = screen_tris[front_facing]

        zbuffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        for tri_idx, (tri_pts, intensity) in enumerate(zip(vis_screen, vis_light)):
            pts = tri_pts.astype(np.int32)
            xmin = max(0, np.min(pts[:,0]))
            xmax = min(self.width-1, np.max(pts[:,0]))
            ymin = max(0, np.min(pts[:,1]))
            ymax = min(self.height-1, np.max(pts[:,1]))
            if xmin > xmax or ymin > ymax:
                continue

            xs_int = np.arange(xmin, xmax+1, dtype=np.int32)
            ys_int = np.arange(ymin, ymax+1, dtype=np.int32)
            px_grid = np.stack(np.meshgrid(xs_int, ys_int), axis=-1).reshape(-1,2)

            v0 = pts[0].astype(np.float32)
            v1 = pts[1].astype(np.float32)
            v2 = pts[2].astype(np.float32)
            v0v1 = v1 - v0
            v0v2 = v2 - v0
            denom = v0v1[0]*v0v2[1] - v0v1[1]*v0v2[0]
            if abs(denom) < 1e-6:
                continue
            inv_denom = 1.0/denom

            px_float = px_grid.astype(np.float32)
            v0p = px_float - v0
            beta = (v0p[:,0]*v0v2[1] - v0p[:,1]*v0v2[0]) * inv_denom
            gamma = (v0v1[0]*v0p[:,1] - v0v1[1]*v0p[:,0]) * inv_denom
            alpha = 1 - beta - gamma

            inside = (alpha>=0) & (beta>=0) & (gamma>=0)
            if not np.any(inside):
                continue

            inside_pixels = px_grid[inside]
            alpha_inside = alpha[inside]
            beta_inside = beta[inside]
            gamma_inside = gamma[inside]

            face_vertex_indices = visible_faces[tri_idx]
            face_z = ndc[face_vertex_indices, 2]
            z_interp = (alpha_inside * face_z[0] + beta_inside * face_z[1] + gamma_inside * face_z[2])

            ys = inside_pixels[:,1].astype(int)
            xs = inside_pixels[:,0].astype(int)
            z_mask = z_interp < zbuffer[ys, xs]
            update_pixels = inside_pixels[z_mask]
            if len(update_pixels) == 0:
                continue

            col = (self.base_color * intensity).astype(np.uint8)
            image[update_pixels[:,1], update_pixels[:,0]] = [col[0], col[1], col[2], 255]
            zbuffer[update_pixels[:,1], update_pixels[:,0]] = z_interp[z_mask]

        return image

# ------------------------------------------------------------
# Animation generation
# ------------------------------------------------------------
def generate_animation(settings, progress_callback, done_callback):
    try:
        mesh = load_model(settings["input_path"])
        vertices = mesh.vertices.copy()
        vertices -= vertices.mean(axis=0)
        scale = 1.0 / np.max(np.linalg.norm(vertices, axis=1))
        vertices *= scale * 1.5

        w, h = settings["width"], settings["height"]
        base_color = settings.get("model_color", [179,179,179])
        renderer = SoftwareRenderer(w, h, base_color)

        total = settings["total_frames"]
        angle_step = settings["orbit_speed"] / total
        frames = []

        distance = 2.5
        elevation = 0.5

        for i in range(total):
            azimuth = np.deg2rad(i * angle_step)
            cam_x = distance * np.cos(azimuth)
            cam_y = elevation * distance
            cam_z = distance * np.sin(azimuth)
            camera_pos = np.array([cam_x, cam_y, cam_z])

            frame = renderer.render(vertices, mesh.faces, camera_pos)
            frames.append(frame)

            percent = int((i+1)/total*100)
            progress_callback(percent)

        writer = imageio.get_writer(
            settings["output_path"], format='FFMPEG', mode='I', fps=30,
            codec='libvpx-vp9', pixelformat='yuva420p',
            output_params=['-crf', '10', '-b:v', '0']
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        done_callback(True, f"WebM saved to:\n{settings['output_path']}")

    except Exception:
        err = traceback.format_exc()
        print(err)
        done_callback(False, err)

# ------------------------------------------------------------
# Dark‑mode GUI (with color picker)
# ------------------------------------------------------------
class OrbitalAnimatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Model Orbital Animation (WebM) – Software Renderer")
        self.root.configure(bg='#2b2b2b')
        self.config = load_config()
        self._create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#444', foreground='white')
        style.map('TButton', background=[('active', '#555')])
        style.configure('TEntry', fieldbackground='#3c3c3c', foreground='white')
        style.configure('TSpinbox', fieldbackground='#3c3c3c', foreground='white')
        style.configure('TProgressbar', background='#4caf50')

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Row 0: Input file
        ttk.Label(main_frame, text="Input Model (STL/3MF):").grid(row=0, column=0, sticky='w', pady=2)
        self.input_var = tk.StringVar(value=self.config["input_path"])
        ttk.Entry(main_frame, textvariable=self.input_var, width=50).grid(row=0, column=1, sticky='we', padx=5)
        ttk.Button(main_frame, text="Browse", command=self._browse_input).grid(row=0, column=2, padx=2)

        # Row 1: Output file
        ttk.Label(main_frame, text="Output WebM:").grid(row=1, column=0, sticky='w', pady=2)
        self.output_var = tk.StringVar(value=self.config["output_path"])
        ttk.Entry(main_frame, textvariable=self.output_var, width=50).grid(row=1, column=1, sticky='we', padx=5)
        ttk.Button(main_frame, text="Browse", command=self._browse_output).grid(row=1, column=2, padx=2)

        # Row 2: Resolution
        ttk.Label(main_frame, text="Width:").grid(row=2, column=0, sticky='e', pady=2)
        self.width_var = tk.IntVar(value=self.config["width"])
        ttk.Spinbox(main_frame, from_=320, to=3840, textvariable=self.width_var, width=8).grid(row=2, column=1, sticky='w', padx=5)
        ttk.Label(main_frame, text="Height:").grid(row=2, column=1, sticky='e', padx=100)
        self.height_var = tk.IntVar(value=self.config["height"])
        ttk.Spinbox(main_frame, from_=240, to=2160, textvariable=self.height_var, width=8).grid(row=2, column=1, sticky='e', padx=5)

        # Row 3: Animation settings
        ttk.Label(main_frame, text="Total Frames:").grid(row=3, column=0, sticky='e', pady=2)
        self.frames_var = tk.IntVar(value=self.config["total_frames"])
        ttk.Spinbox(main_frame, from_=10, to=1000, textvariable=self.frames_var, width=8).grid(row=3, column=1, sticky='w', padx=5)

        ttk.Label(main_frame, text="Orbit Speed (deg):").grid(row=3, column=1, sticky='e', padx=60)
        self.speed_var = tk.DoubleVar(value=self.config["orbit_speed"])
        ttk.Spinbox(main_frame, from_=10, to=3600, textvariable=self.speed_var, width=8).grid(row=3, column=1, sticky='e', padx=5)

        # Row 4: Model color picker
        ttk.Label(main_frame, text="Model Color:").grid(row=4, column=0, sticky='e', pady=5)

        model_color = self.config.get("model_color", [179, 179, 179])   # safe fallback
        self.color_btn = tk.Button(
            main_frame, text="", bg=self._rgb_to_hex(model_color),
            activebackground=self._rgb_to_hex(model_color),
            command=self._pick_color, relief=tk.FLAT, width=4, height=1, borderwidth=1
        )
        self.color_btn.grid(row=4, column=1, sticky='w', padx=5)
        self.color_label = ttk.Label(
            main_frame,
            text=self._rgb_to_text(model_color),
            foreground='#aaaaaa'
        )
        self.color_label.grid(row=4, column=1, sticky='e', padx=120)

        # Row 5: Generate button
        self.generate_btn = ttk.Button(main_frame, text="Generate WebM", command=self._start_generation)
        self.generate_btn.grid(row=5, column=0, columnspan=3, pady=10)

        # Row 6: Progress bar
        self.progress = ttk.Progressbar(main_frame, orient='horizontal', length=400, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=3, pady=5, sticky='we')

        # Row 7: Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, foreground='#aaaaaa').grid(row=7, column=0, columnspan=3)

        main_frame.columnconfigure(1, weight=1)

    def _rgb_to_hex(self, rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _rgb_to_text(self, rgb):
        return f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})"

    def _pick_color(self):
        color_tuple = colorchooser.askcolor(
            color=self._rgb_to_hex(self.config.get("model_color", [179,179,179])),
            title="Choose Model Color"
        )
        if color_tuple[0] is not None:
            r, g, b = [int(c) for c in color_tuple[0]]
            self.config["model_color"] = [r, g, b]
            self.color_btn.config(bg=self._rgb_to_hex([r,g,b]))
            self.color_label.config(text=self._rgb_to_text([r,g,b]))
            save_config(self.config)

    def _browse_input(self):
        path = filedialog.askopenfilename(title="Select 3D Model", filetypes=[("3D Models", "*.stl *.3mf"), ("All files", "*.*")])
        if path: self.input_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="Save WebM As", defaultextension=".webm", filetypes=[("WebM video", "*.webm")])
        if path: self.output_var.set(path)

    def _start_generation(self):
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Invalid input file.")
            return
        if not output_path:
            messagebox.showerror("Error", "Please specify an output file.")
            return

        self.config.update({
            "input_path": input_path,
            "output_path": output_path,
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "total_frames": self.frames_var.get(),
            "orbit_speed": self.speed_var.get()
            # model_color already in config
        })
        save_config(self.config)

        self.generate_btn.configure(state="disabled")
        self.status_var.set("Rendering (CPU) …")
        self.progress["value"] = 0

        thread = threading.Thread(target=generate_animation,
                                  args=(self.config, self._update_progress, self._generation_done))
        thread.daemon = True
        thread.start()

    def _update_progress(self, percent):
        self.root.after(0, lambda: self.progress.configure(value=percent))

    def _generation_done(self, success, message):
        def _cb():
            self.generate_btn.configure(state="normal")
            if success:
                self.status_var.set("Finished")
                messagebox.showinfo("Success", message)
            else:
                self.status_var.set("Error")
                msg_box = tk.Toplevel(self.root)
                msg_box.title("Error Details")
                msg_box.configure(bg='#2b2b2b')
                text = tk.Text(msg_box, bg='#3c3c3c', fg='white', wrap=tk.WORD, width=80, height=15)
                text.insert(tk.END, message)
                text.config(state=tk.DISABLED)
                text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
                ttk.Button(msg_box, text="Close", command=msg_box.destroy).pack(pady=5)
        self.root.after(0, _cb)

    def _on_closing(self):
        save_config(self.config)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OrbitalAnimatorApp(root)
    root.mainloop()