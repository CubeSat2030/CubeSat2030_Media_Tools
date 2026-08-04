import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import subprocess
import shutil

class TimelapseCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Timelapse Creator")
        self.root.geometry("620x550")
        self.root.resizable(True, True)

        # Variables
        self.input_mode = tk.StringVar(value="folder")   # "folder" or "files"
        self.input_folder = tk.StringVar()
        self.file_list = []                              # list of file paths
        self.output_file = tk.StringVar(value="timelapse.mp4")
        self.fps = tk.IntVar(value=24)
        self.thumb_path = tk.StringVar()
        self.thumb_duration = tk.DoubleVar(value=2.0)
        self.use_ffmpeg = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready")
        self.ffmpeg_available = self.check_ffmpeg()

        # --- Mode selection ---
        mode_frame = tk.LabelFrame(root, text="Input Mode", padx=5, pady=5)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        tk.Radiobutton(mode_frame, text="Select Folder (all images inside)", variable=self.input_mode,
                       value="folder", command=self.switch_mode).pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Select Individual Files", variable=self.input_mode,
                       value="files", command=self.switch_mode).pack(side="left", padx=5)

        # --- Folder selection (row 1) ---
        self.folder_frame = tk.Frame(root)
        self.folder_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        tk.Label(self.folder_frame, text="Input Folder:").pack(side="left")
        tk.Entry(self.folder_frame, textvariable=self.input_folder, width=40).pack(side="left", padx=5)
        tk.Button(self.folder_frame, text="Browse", command=self.browse_folder).pack(side="left")

        # --- File list (row 2 - 4) ---
        self.files_frame = tk.LabelFrame(root, text="Selected Files (drag not supported, use buttons)", padx=5, pady=5)
        self.files_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=5)
        self.files_listbox = tk.Listbox(self.files_frame, selectmode=tk.SINGLE, height=6)
        self.files_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(self.files_frame, orient="vertical", command=self.files_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.files_listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = tk.Frame(self.files_frame)
        btn_frame.pack(side="bottom", fill="x", pady=5)
        tk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Move Up", command=self.move_up).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Move Down", command=self.move_down).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Clear All", command=self.clear_files).pack(side="left", padx=2)

        # --- Output file (row 5) ---
        out_frame = tk.Frame(root)
        out_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        tk.Label(out_frame, text="Output Video:").pack(side="left")
        tk.Entry(out_frame, textvariable=self.output_file, width=40).pack(side="left", padx=5)
        tk.Button(out_frame, text="Browse", command=self.browse_output).pack(side="left")

        # --- FPS (row 6) ---
        fps_frame = tk.Frame(root)
        fps_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        tk.Label(fps_frame, text="FPS:").pack(side="left")
        tk.Spinbox(fps_frame, from_=1, to=120, textvariable=self.fps, width=5).pack(side="left", padx=5)

        # --- Thumbnail (row 7-8) ---
        thumb_frame = tk.Frame(root)
        thumb_frame.grid(row=7, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        tk.Label(thumb_frame, text="Thumbnail Image:").pack(side="left")
        tk.Entry(thumb_frame, textvariable=self.thumb_path, width=40).pack(side="left", padx=5)
        tk.Button(thumb_frame, text="Browse", command=self.browse_thumb).pack(side="left")

        dur_frame = tk.Frame(root)
        dur_frame.grid(row=8, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        tk.Label(dur_frame, text="Still frame duration (sec):").pack(side="left")
        tk.Spinbox(dur_frame, from_=0.5, to=10.0, increment=0.5, textvariable=self.thumb_duration, width=5).pack(side="left", padx=5)

        # --- FFmpeg checkbox (row 9) ---
        self.ffmpeg_check = tk.Checkbutton(root, text="Embed as metadata (requires FFmpeg)",
                                            variable=self.use_ffmpeg,
                                            state="normal" if self.ffmpeg_available else "disabled")
        self.ffmpeg_check.grid(row=9, column=1, pady=5, sticky="w")
        if not self.ffmpeg_available:
            self.use_ffmpeg.set(False)
            tk.Label(root, text="⚠ FFmpeg not found – will insert thumbnail as still frame",
                     fg="red", font=("Arial", 8)).grid(row=10, column=1, sticky="w")

        # --- Create button ---
        tk.Button(root, text="Create Timelapse", command=self.create_timelapse,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).grid(row=11, column=1, pady=15)

        # Status
        tk.Label(root, textvariable=self.status, fg="blue").grid(row=12, column=0, columnspan=3)

        # Initial mode setup
        self.switch_mode()

        # Make rows/cols expandable
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(1, weight=1)

    def check_ffmpeg(self):
        return shutil.which("ffmpeg") is not None

    def switch_mode(self):
        """Show/hide controls based on selected mode."""
        if self.input_mode.get() == "folder":
            self.folder_frame.grid()
            self.files_frame.grid_remove()
        else:
            self.folder_frame.grid_remove()
            self.files_frame.grid()

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if folder:
            self.input_folder.set(folder)

    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")]
        )
        if files:
            for f in files:
                if f not in self.file_list:
                    self.file_list.append(f)
                    self.files_listbox.insert(tk.END, os.path.basename(f))

    def remove_selected(self):
        sel = self.files_listbox.curselection()
        if sel:
            idx = sel[0]
            del self.file_list[idx]
            self.files_listbox.delete(idx)

    def move_up(self):
        sel = self.files_listbox.curselection()
        if sel and sel[0] > 0:
            idx = sel[0]
            # Swap in list
            self.file_list[idx], self.file_list[idx-1] = self.file_list[idx-1], self.file_list[idx]
            # Refresh listbox
            self.refresh_listbox()
            self.files_listbox.selection_set(idx-1)

    def move_down(self):
        sel = self.files_listbox.curselection()
        if sel is not None and sel[0] < len(self.file_list)-1:
            idx = sel[0]
            self.file_list[idx], self.file_list[idx+1] = self.file_list[idx+1], self.file_list[idx]
            self.refresh_listbox()
            self.files_listbox.selection_set(idx+1)

    def clear_files(self):
        self.file_list.clear()
        self.files_listbox.delete(0, tk.END)

    def refresh_listbox(self):
        self.files_listbox.delete(0, tk.END)
        for f in self.file_list:
            self.files_listbox.insert(tk.END, os.path.basename(f))

    def browse_output(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov"), ("All files", "*.*")]
        )
        if file:
            self.output_file.set(file)

    def browse_thumb(self):
        file = filedialog.askopenfilename(
            title="Select thumbnail image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file:
            self.thumb_path.set(file)

    def get_image_list(self):
        """Return a sorted list of image paths based on current mode."""
        if self.input_mode.get() == "folder":
            folder = self.input_folder.get()
            if not folder:
                return None
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            images = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(valid_exts)]
            images.sort()
            return images
        else:
            if not self.file_list:
                return None
            return self.file_list[:]   # already ordered by user

    def create_timelapse(self):
        images = self.get_image_list()
        if not images:
            messagebox.showerror("Error", "No images selected. Please choose a folder or add files.")
            return

        out_path = self.output_file.get()
        fps = self.fps.get()
        thumb = self.thumb_path.get()
        duration = self.thumb_duration.get()
        embed_meta = self.use_ffmpeg.get() and self.ffmpeg_available

        # Read first image to get dimensions
        first_frame = cv2.imread(images[0])
        if first_frame is None:
            messagebox.showerror("Error", "Could not read the first image.")
            return
        height, width, _ = first_frame.shape

        use_temp = embed_meta and bool(thumb)
        if use_temp:
            temp_path = "temp_timelapse.avi"
        else:
            temp_path = out_path
            if out_path.lower().endswith('.mp4'):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif out_path.lower().endswith('.mov'):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')

        thumb_frame = None
        if not embed_meta and thumb:
            if not os.path.isfile(thumb):
                messagebox.showerror("Error", "Thumbnail file not found.")
                return
            thumb_img = cv2.imread(thumb)
            if thumb_img is None:
                messagebox.showerror("Error", "Could not read the thumbnail image.")
                return
            thumb_frame = cv2.resize(thumb_img, (width, height))
            thumb_frames_count = int(fps * duration)

        if use_temp:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        # Write thumbnail still frames if fallback
        if thumb_frame is not None:
            self.status.set("Writing thumbnail still frames...")
            self.root.update()
            for i in range(thumb_frames_count):
                video.write(thumb_frame)
                if i % fps == 0:
                    self.status.set(f"Writing thumbnail frames ({i+1}/{thumb_frames_count})")
                    self.root.update()

        # Write timelapse images
        total = len(images)
        for i, img_path in enumerate(images, 1):
            self.status.set(f"Processing image {i}/{total}: {os.path.basename(img_path)}")
            self.root.update()
            img = cv2.imread(img_path)
            if img is None:
                continue
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            video.write(img)

        video.release()

        if use_temp:
            self.status.set("Embedding thumbnail as metadata with FFmpeg...")
            self.root.update()
            success = self.embed_thumbnail_ffmpeg(temp_path, thumb, out_path)
            try:
                os.remove(temp_path)
            except:
                pass
            if success:
                self.status.set("Done! Timelapse with embedded thumbnail created.")
                messagebox.showinfo("Success", f"Timelapse saved to:\n{out_path}")
            else:
                self.status.set("FFmpeg embedding failed. Temp video kept.")
                messagebox.showerror("Error", "FFmpeg embedding failed. Check console for details.")
        else:
            self.status.set("Done! Timelapse created.")
            messagebox.showinfo("Success", f"Timelapse saved to:\n{out_path}")

    def embed_thumbnail_ffmpeg(self, input_video, thumb_image, output_video):
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-i", thumb_image,
            "-map", "0:v:0",
            "-map", "1",
            "-c:v", "copy",
            "-c:a", "copy",
            "-disposition:v:1", "attached_pic",
            output_video
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print("FFmpeg error:", proc.stderr)
                return False
            return True
        except Exception as e:
            print("FFmpeg call failed:", e)
            return False

if __name__ == "__main__":
    root = tk.Tk()
    app = TimelapseCreator(root)
    root.mainloop()