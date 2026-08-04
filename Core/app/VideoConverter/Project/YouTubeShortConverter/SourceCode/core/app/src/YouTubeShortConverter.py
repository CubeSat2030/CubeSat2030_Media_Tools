import os
import sys
import json
import subprocess
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

CONFIG_FILE = "ytshorts_converter_config.json"
FFMPEG_CMD = "ffmpeg"

LOSSY_SETTINGS = {
    "Lossless H.264 (libx264)": {
        "vcodec": "libx264",
        "extra": ["-crf", "0", "-preset", "ultrafast"]
    },
    "Lossless H.265 (libx265)": {
        "vcodec": "libx265",
        "extra": ["-x265-params", "lossless=1"]
    },
    "Lossless FFV1": {
        "vcodec": "ffv1",
        "extra": ["-level", "3", "-coder", "1", "-context", "1"]
    }
}

FIT_MODES = ["Stretch", "Crop", "Pad"]
SPEED_VALUES = ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0", "3.0", "4.0"]

class YouTubeShortConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTubeShort Converter")
        self.root.resizable(False, False)

        # State variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_name = tk.StringVar(value="output")
        self.width_var = tk.IntVar(value=1080)
        self.height_var = tk.IntVar(value=1920)
        self.fit_mode = tk.StringVar(value=FIT_MODES[1])  # Crop
        self.rotate_var = tk.IntVar(value=0)
        self.speed_var = tk.StringVar(value="1.0")
        self.lossy_codec = tk.StringVar(value=list(LOSSY_SETTINGS.keys())[0])
        self.preview_fast = tk.BooleanVar(value=True)   # Use fast lossy for preview
        self.status_text = tk.StringVar(value="Ready")
        self.preview_status = tk.StringVar(value="No preview")
        self.is_running = False

        # Load saved paths
        self.config = self.load_config()

        # Build UI
        self.create_widgets()

        # Apply saved directories
        if self.config.get("last_input_dir"):
            self.input_file.set("")
        if self.config.get("last_output_dir"):
            self.output_dir.set(self.config["last_output_dir"])

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- Config management ----
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_config(self):
        config = {
            "last_input_dir": os.path.dirname(self.input_file.get()) if self.input_file.get() else "",
            "last_output_dir": self.output_dir.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
        except:
            pass

    def on_close(self):
        self.save_config()
        self.root.destroy()

    # ---- UI construction ----
    def create_widgets(self):
        pad = 10

        # ----- Input file -----
        frm_input = ttk.LabelFrame(self.root, text="Input Video", padding=pad)
        frm_input.pack(fill="x", padx=pad, pady=(pad, 0))

        ttk.Entry(frm_input, textvariable=self.input_file, width=60).pack(side="left", fill="x", expand=True, padx=(0, pad))
        ttk.Button(frm_input, text="Browse", command=self.browse_input).pack(side="right")

        # ----- Output folder & name -----
        frm_output = ttk.LabelFrame(self.root, text="Output Settings", padding=pad)
        frm_output.pack(fill="x", padx=pad, pady=(pad, 0))

        ttk.Label(frm_output, text="Folder:").grid(row=0, column=0, sticky="w", padx=(0,5))
        ttk.Entry(frm_output, textvariable=self.output_dir, width=50).grid(row=0, column=1, sticky="ew")
        ttk.Button(frm_output, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=(5,0))

        ttk.Label(frm_output, text="File name (without ext.):").grid(row=1, column=0, sticky="w", pady=(5,0))
        ttk.Entry(frm_output, textvariable=self.output_name, width=30).grid(row=1, column=1, sticky="w", pady=(5,0))
        frm_output.columnconfigure(1, weight=1)

        # ----- Resolution and Fit -----
        frm_res = ttk.LabelFrame(self.root, text="Resolution & Aspect Fit", padding=pad)
        frm_res.pack(fill="x", padx=pad, pady=(pad, 0))

        ttk.Label(frm_res, text="Width:").grid(row=0, column=0, padx=(0,5), pady=5)
        ttk.Entry(frm_res, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=(0,15))
        ttk.Label(frm_res, text="Height:").grid(row=0, column=2, padx=(0,5), pady=5)
        ttk.Entry(frm_res, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=(0,15))
        ttk.Button(frm_res, text="1080×1920", command=lambda: self.set_res(1080,1920)).grid(row=0, column=4, padx=2)
        ttk.Button(frm_res, text="720×1280", command=lambda: self.set_res(720,1280)).grid(row=0, column=5, padx=2)
        ttk.Button(frm_res, text="1920×1080", command=lambda: self.set_res(1920,1080)).grid(row=0, column=6, padx=2)

        ttk.Label(frm_res, text="Fit mode:").grid(row=1, column=0, padx=(0,5), pady=5, sticky="w")
        cmb_fit = ttk.Combobox(frm_res, textvariable=self.fit_mode, values=FIT_MODES, state="readonly", width=12)
        cmb_fit.grid(row=1, column=1, columnspan=2, sticky="w")

        fit_explanations = {
            "Stretch": "Distort to exactly fit W×H",
            "Crop": "Resize keeping ratio, then crop edges",
            "Pad": "Resize keeping ratio, add black bars"
        }
        self.lbl_fit_info = ttk.Label(frm_res, text=fit_explanations["Crop"], foreground="gray")
        self.lbl_fit_info.grid(row=2, column=0, columnspan=7, sticky="w", pady=(0,5))
        cmb_fit.bind("<<ComboboxSelected>>", lambda e: self.lbl_fit_info.config(text=fit_explanations[self.fit_mode.get()]))

        # ----- Rotation -----
        frm_rot = ttk.LabelFrame(self.root, text="Rotation", padding=pad)
        frm_rot.pack(fill="x", padx=pad, pady=(pad, 0))
        ttk.Radiobutton(frm_rot, text="0°", variable=self.rotate_var, value=0).pack(side="left")
        ttk.Radiobutton(frm_rot, text="90°", variable=self.rotate_var, value=90).pack(side="left", padx=(10,0))
        ttk.Radiobutton(frm_rot, text="180°", variable=self.rotate_var, value=180).pack(side="left", padx=(10,0))
        ttk.Radiobutton(frm_rot, text="270°", variable=self.rotate_var, value=270).pack(side="left", padx=(10,0))

        # ----- Speed -----
        frm_speed = ttk.LabelFrame(self.root, text="Speed", padding=pad)
        frm_speed.pack(fill="x", padx=pad, pady=(pad, 0))
        ttk.Label(frm_speed, text="Playback speed:").pack(side="left", padx=(0,5))
        cmb_speed = ttk.Combobox(frm_speed, textvariable=self.speed_var, values=SPEED_VALUES, state="readonly", width=8)
        cmb_speed.pack(side="left")
        ttk.Label(frm_speed, text="×  (audio pitch corrected)", foreground="gray").pack(side="left", padx=(5,0))

        # ----- Compression -----
        frm_codec = ttk.LabelFrame(self.root, text="Lossless Compression", padding=pad)
        frm_codec.pack(fill="x", padx=pad, pady=(pad, 0))
        cmb_codec = ttk.Combobox(frm_codec, textvariable=self.lossy_codec,
                                values=list(LOSSY_SETTINGS.keys()), state="readonly", width=40)
        cmb_codec.pack(anchor="w")
        ttk.Label(frm_codec, text="(lossless encoding – mathematically identical, files can still be large)",
                  foreground="gray").pack(anchor="w", pady=(2,0))

        # ----- Preview display box -----
        frm_display = ttk.LabelFrame(self.root, text="Preview Display", padding=pad)
        frm_display.pack(fill="both", padx=pad, pady=(pad,0))

        self.display_canvas = tk.Canvas(frm_display, width=480, height=270, bg="black", highlightthickness=0)
        self.display_canvas.pack(pady=(0,5))
        self.display_canvas.create_text(240, 135, text="Preview not generated", fill="white", font=("Arial", 14))

        ttk.Checkbutton(frm_display, text="Use fast lossy encode for preview (much quicker)",
                       variable=self.preview_fast).pack(anchor="w", pady=(0,2))
        ttk.Label(frm_display, textvariable=self.preview_status, foreground="gray").pack(anchor="w")

        # ----- Convert and Preview buttons, status -----
        frm_action = ttk.Frame(self.root, padding=pad)
        frm_action.pack(fill="x", padx=pad, pady=pad)
        self.btn_convert = ttk.Button(frm_action, text="Convert", command=self.start_conversion)
        self.btn_convert.pack(side="left", padx=(0, 10))
        self.btn_preview = ttk.Button(frm_action, text="Preview (5 sec sample)", command=self.start_preview)
        self.btn_preview.pack(side="left", padx=(0, 20))
        ttk.Label(frm_action, textvariable=self.status_text, foreground="blue").pack(side="left", fill="x", expand=True)

    def set_res(self, w, h):
        self.width_var.set(w)
        self.height_var.set(h)

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.flv"), ("All files", "*.*")]
        )
        if path:
            self.input_file.set(path)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(path))
            # Update preview status
            self.preview_status.set("Input file selected.")

    def browse_output_dir(self):
        dir_path = filedialog.askdirectory(title="Select output folder", mustexist=True)
        if dir_path:
            self.output_dir.set(dir_path)

    # ---- Validation ----
    def validate(self, for_preview=False):
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input video.")
            return False
        if not os.path.isfile(self.input_file.get()):
            messagebox.showerror("Error", "Input video does not exist.")
            return False
        if not for_preview:
            if not self.output_dir.get():
                messagebox.showerror("Error", "Please choose an output folder.")
                return False
            if not os.path.isdir(self.output_dir.get()):
                messagebox.showerror("Error", "Output folder does not exist.")
                return False
            if not self.output_name.get().strip():
                messagebox.showerror("Error", "Please enter an output file name.")
                return False
        try:
            w = self.width_var.get()
            h = self.height_var.get()
            if w <= 0 or h <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Width and height must be positive integers.")
            return False
        try:
            float(self.speed_var.get())
        except:
            messagebox.showerror("Error", "Invalid speed value.")
            return False
        return True

    # ---- Build FFmpeg command (used for both full conversion and preview) ----
    def build_command(self, out_path, preview_duration=None, fast_preview=False):
        w, h = self.width_var.get(), self.height_var.get()
        fit = self.fit_mode.get()
        rotate = self.rotate_var.get()
        speed = float(self.speed_var.get())

        # Scale and fit filter
        if fit == "Stretch":
            scale_filter = f"scale={w}:{h}"
        elif fit == "Crop":
            scale_filter = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        else:  # Pad
            scale_filter = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

        # Rotation filter
        if rotate == 90:
            rot_filter = "transpose=1"
        elif rotate == 180:
            rot_filter = "transpose=1,transpose=1"
        elif rotate == 270:
            rot_filter = "transpose=2"
        else:
            rot_filter = None

        # Speed filter (video)
        if speed != 1.0:
            speed_vf = f"setpts={1/speed}*PTS"
        else:
            speed_vf = None

        # Combine all video filters
        vf_parts = [scale_filter]
        if rot_filter:
            vf_parts.append(rot_filter)
        if speed_vf:
            vf_parts.append(speed_vf)
        filter_complex = ",".join(vf_parts)

        # Audio filters (atempo)
        af_parts = []
        if speed != 1.0:
            atempo_chain = self.build_atempo_chain(speed)
            af_parts.append(atempo_chain)
        audio_filters = ",".join(af_parts) if af_parts else None

        # Determine codec settings
        if fast_preview:
            vcodec = "libx264"
            vcodec_extra = ["-crf", "23", "-preset", "veryfast"]
            acodec = "aac"   # re-encode audio for preview compatibility
        else:
            codec_info = LOSSY_SETTINGS[self.lossy_codec.get()]
            vcodec = codec_info["vcodec"]
            vcodec_extra = codec_info["extra"]
            acodec = "copy"  # keep original audio (lossless)

        cmd = [
            FFMPEG_CMD,
            "-y",
            "-i", self.input_file.get(),
            "-vf", filter_complex,
        ]

        if audio_filters:
            cmd += ["-af", audio_filters]

        cmd += [
            "-c:v", vcodec,
            *vcodec_extra,
            "-c:a", acodec,
        ]

        if preview_duration is not None:
            cmd += ["-t", str(preview_duration)]

        cmd.append(out_path)
        return cmd

    # ---- Build atempo chain for arbitrary speed ----
    @staticmethod
    def build_atempo_chain(speed):
        """
        atempo filter only accepts values between 0.5 and 2.0.
        Chain multiple atempo filters to achieve any speed.
        """
        if speed == 1.0:
            return "atempo=1.0"
        remaining = speed
        chain = []
        while remaining > 2.0:
            chain.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            chain.append("atempo=0.5")
            remaining /= 0.5
        chain.append(f"atempo={remaining:.4f}")
        return ",".join(chain)

    # ---- Preview ----
    def start_preview(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return
        if not self.validate(for_preview=True):
            return

        # Create a temporary file for the preview
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.close()

        self.is_running = True
        self.btn_preview.config(state="disabled")
        self.btn_convert.config(state="disabled")
        self.status_text.set("Generating preview...")
        self.preview_status.set("Rendering 5‑second preview...")

        thread = threading.Thread(target=self.run_preview, args=(tmp_path,), daemon=True)
        thread.start()

    def run_preview(self, out_path):
        cmd = self.build_command(out_path, preview_duration=5, fast_preview=self.preview_fast.get())
        print("Preview command:", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                self.root.after(0, self.preview_finished, True, out_path)
            else:
                error_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr else "Unknown error"
                self.root.after(0, self.preview_finished, False, f"FFmpeg error (code {proc.returncode}): {error_msg}")
        except FileNotFoundError:
            self.root.after(0, self.preview_finished, False, "ffmpeg not found.")
        except Exception as e:
            self.root.after(0, self.preview_finished, False, str(e))

    def preview_finished(self, success, path):
        self.is_running = False
        self.btn_preview.config(state="normal")
        self.btn_convert.config(state="normal")
        if success:
            self.status_text.set("Preview ready")
            self.preview_status.set(f"Preview saved: {os.path.basename(path)}")
            # Update the display box text
            self.display_canvas.delete("all")
            self.display_canvas.create_text(240, 135, text="Preview generated!", fill="lime", font=("Arial", 14))
            # Open the file with the system’s default video player
            self.open_file(path)
        else:
            self.status_text.set("Preview failed")
            self.preview_status.set("Error generating preview")
            self.display_canvas.delete("all")
            self.display_canvas.create_text(240, 135, text="Preview failed", fill="red", font=("Arial", 14))
            messagebox.showerror("Preview Error", str(path))

    def open_file(self, filepath):
        """Open a file with the default OS application."""
        try:
            if sys.platform.startswith("win"):
                os.startfile(filepath)
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            messagebox.showwarning("Open Error", f"Could not open preview:\n{e}")

    # ---- Full conversion (unchanged except using new build_command) ----
    def start_conversion(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return
        if not self.validate():
            return

        out_name = self.output_name.get().strip()
        out_dir = self.output_dir.get()
        out_path = os.path.join(out_dir, f"{out_name}.mkv")

        self.is_running = True
        self.btn_convert.config(state="disabled")
        self.btn_preview.config(state="disabled")
        self.status_text.set("Converting...")

        thread = threading.Thread(target=self.run_conversion, args=(out_path,), daemon=True)
        thread.start()

    def run_conversion(self, out_path):
        cmd = self.build_command(out_path, fast_preview=False)  # always lossless
        print("Conversion command:", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                self.root.after(0, self.conversion_finished, True, out_path)
            else:
                error_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr else "Unknown error"
                self.root.after(0, self.conversion_finished, False, f"FFmpeg error (code {proc.returncode}): {error_msg}")
        except FileNotFoundError:
            self.root.after(0, self.conversion_finished, False, "ffmpeg not found.")
        except Exception as e:
            self.root.after(0, self.conversion_finished, False, str(e))

    def conversion_finished(self, success, message):
        self.is_running = False
        self.btn_convert.config(state="normal")
        self.btn_preview.config(state="normal")
        if success:
            self.status_text.set(f"Done! Saved to {message}")
            messagebox.showinfo("Success", f"Video saved successfully:\n{message}")
        else:
            self.status_text.set("Error")
            messagebox.showerror("Conversion failed", str(message))
        self.save_config()


if __name__ == "__main__":
    # Check for ffmpeg
    try:
        subprocess.run([FFMPEG_CMD, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing Dependency",
                             "FFmpeg is not installed or not found in PATH.\n"
                             "Please install FFmpeg and try again.")
        sys.exit(1)

    root = tk.Tk()
    app = YouTubeShortConverter(root)
    root.mainloop()