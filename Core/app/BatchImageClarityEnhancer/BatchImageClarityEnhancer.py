import torch
import torch.nn as nn
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import time
import queue as q_temp
import threading
import cv2 
import math

# --- GLOBAL SETTINGS ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCALE_FACTOR = 4 

# --- MEMORY FIX: MAX INPUT SIZE FOR 10GB RAM ---
# Set to 0.6 for highest detail input while trying to avoid OOM.
PRE_RESIZE_SCALE = 0.6 

# --- BICUBIC FALLBACK THRESHOLD (STRICT) ---
ARTIFACT_THRESHOLD = 15 

# ======================================================================
# 🚀 EDSR ARCHITECTURE (32 BLOCKS, 256 CHANNELS)
# ======================================================================

class MeanShift(nn.Conv2d):
    """Handles the EDSR custom normalization (mean subtraction/addition)."""
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) 
        for params in self.parameters():
            params.requires_grad = False

class ResidualBlock(nn.Module):
    """The fundamental residual block used in EDSR."""
    def __init__(self, n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResidualBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR_PyTorch_Model(nn.Module):
    """The full EDSR network, configured for the larger 32-block, 256-channel variant."""
    def __init__(self, scale=4, n_resblocks=32, n_feats=256, rgb_range=1.0):
        super(EDSR_PyTorch_Model, self).__init__()
        
        n_resblocks = 32
        n_feats = 256
        rgb_mean = (0.4488, 0.4371, 0.4040)
        
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.conv_input = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)

        res_blocks = [ResidualBlock(n_feats) for _ in range(n_resblocks)]
        res_blocks.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*res_blocks)

        m_up = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                m_up.append(nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1))
                m_up.append(nn.PixelShuffle(2))
        elif scale == 3: 
            m_up.append(nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, padding=1))
            m_up.append(nn.PixelShuffle(3))
        else:
             m_up.append(nn.Upsample(scale_factor=scale, mode='nearest'))
             m_up.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))

        m_up.append(nn.Conv2d(n_feats, 3, kernel_size=3, padding=1))
        self.upsample = nn.Sequential(*m_up)
        self.add_mean = MeanShift(rgb_mean, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.conv_input(x)
        res = self.body(x)
        res += x
        x = self.upsample(res)
        x = self.add_mean(x)
        return x

# ======================================================================
# --- GUI AND CORE LOGIC ---
# ======================================================================
class BatchImageClarityEnhancer:
    def __init__(self, master):
        self.master = master
        master.title("Batch Image Clarity Enhancer (EDSR PyTorch Fixed)")
        master.geometry("800x600")
        master.configure(bg="#2c3e50")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready. Select directories and PyTorch model.")

        self.is_running = False
        self.job_queue = []
        self.total_frames = 0
        self.processed_count = 0
        
        self.progress_queue = q_temp.Queue()
        self.stop_flag = threading.Event()

        self.create_widgets()

    # --- GUI METHODS (omitted for brevity) ---
    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#34495e')
        style.configure('TLabel', background='#2c3e50', foreground='white', font=('Inter', 10))
        style.configure('TButton', font=('Inter', 10, 'bold'), padding=6, background='#1abc9c', foreground='white')
        style.map('TButton', background=[('active', '#16a085')])
        style.configure('TProgressbar', thickness=15, troughcolor='#34495e', background='#2ecc71')

        main_frame = ttk.Frame(self.master, padding="15", style='TFrame')
        main_frame.pack(fill='both', expand=True)

        title_label = ttk.Label(main_frame, text="Super Resolution GPU Processor (EDSR Fixed)", font=('Inter', 18, 'bold'), anchor='center')
        title_label.pack(pady=(5, 20))

        dir_frame = ttk.Frame(main_frame, style='TFrame')
        dir_frame.pack(fill='x', pady=5)

        self.create_path_selector(dir_frame, "Input Frames Directory:", self.input_dir, self.select_input_dir)
        self.create_path_selector(dir_frame, "Output Directory:", self.output_dir, self.select_output_dir)
        self.create_path_selector(dir_frame, "Model (.pt) File:", self.model_path, self.select_model_file)

        status_label = ttk.Label(main_frame, text="Status:", style='TLabel')
        status_label.pack(fill='x', pady=(10, 0))

        status_display = ttk.Label(main_frame, textvariable=self.status_var, background='#34495e', foreground='#f1c40f', font=('Inter', 10, 'italic'), padding=5)
        status_display.pack(fill='x', pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate", style='TProgressbar')
        self.progress_bar.pack(fill='x', pady=10)

        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.pack(fill='x', pady=20)

        self.start_button = ttk.Button(button_frame, text="Start GPU Enhancement", command=self.start_enhancement, style='TButton')
        self.start_button.pack(side='left', expand=True, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_enhancement, state='disabled', style='TButton')
        self.stop_button.pack(side='left', expand=True, padx=10)

        log_label = ttk.Label(main_frame, text="Processing Log:", style='TLabel')
        log_label.pack(fill='x', pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(main_frame, height=8, state='disabled', bg='#1f2b37', fg='#ecf0f1', font=('Inter', 9), relief=tk.FLAT)
        self.log_text.pack(fill='both', expand=True)

        core_info = ttk.Label(main_frame, text=f"Configured for Single-Threaded GPU Processing on Device: {DEVICE}", background='#2c3e50', foreground='#95a5a6', font=('Inter', 8))
        core_info.pack(pady=5)
    
    def create_path_selector(self, parent, label_text, var, command):
        frame = ttk.Frame(parent, style='TFrame')
        frame.pack(fill='x', pady=5)

        label = ttk.Label(frame, text=label_text, width=25)
        label.pack(side='left', padx=(0, 10))

        entry = ttk.Entry(frame, textvariable=var, state='readonly', width=60)
        entry.pack(side='left', fill='x', expand=True)

        button = ttk.Button(frame, text="Browse", command=command)
        button.pack(side='right', padx=5)

    def log(self, message):
        self.master.after(0, self._append_log, message)

    def _append_log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.yview(tk.END)
        self.log_text.config(state='disabled')

    def select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Directory with Input Frames")
        if directory:
            self.input_dir.set(directory)

    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory for Enhanced Frames")
        if directory:
            self.output_dir.set(directory)

    def select_model_file(self):
        file = filedialog.askopenfilename(title="Select PyTorch Super Resolution Model (.pth/.pt)", filetypes=[("PyTorch Models", "*.pt *.pth")])
        if file:
            self.model_path.set(file)

    def start_enhancement(self):
        if self.is_running:
            self.status_var.set("Already running...")
            return

        input_path = self.input_dir.get()
        output_path = self.output_dir.get()
        model_file = self.model_path.get()
        
        if not all([input_path, output_path, model_file]):
            self.status_var.set("Error: All paths (Input, Output, Model) must be selected.")
            return

        if not os.path.exists(model_file):
            self.status_var.set(f"Error: Model file not found at {model_file}")
            return

        image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            self.status_var.set("Error: No images found in the input directory.")
            return

        # Note: output_path is still used for the base directory, the file extension is fixed inside the loop.
        self.job_queue = [(os.path.join(input_path, f), os.path.join(output_path, f)) for f in image_files]
        self.total_frames = len(self.job_queue)
        self.processed_count = 0
        self.stop_flag.clear()
        self.is_running = True
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')

        self.status_var.set(f"Starting PyTorch GPU enhancement on {self.total_frames} frames...")
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = self.total_frames

        threading.Thread(target=self._run_gpu_processing_loop, daemon=True).start()
        self.master.after(100, self.update_progress)

    def _check_artifacting(self, img_np_bgr):
        """
        Checks if the EDSR output image is dominated by excessive noise/artifacts.
        This is done by calculating the standard deviation of the image, 
        which is a good measure of high-frequency noise/texture.
        """
        if img_np_bgr.size == 0:
            return True # Treat empty image as artifacted
        
        # Calculate the standard deviation (noise) across the entire BGR image
        std_dev = np.std(img_np_bgr)
        
        # Lower threshold means stricter check, forcing more to Bicubic
        return std_dev > ARTIFACT_THRESHOLD
        
    def _run_gpu_processing_loop(self):
        with torch.no_grad():
            try:
                # 1. CHECK DEVICE AVAILABILITY & RAM
                if DEVICE.type == 'cpu':
                    self.progress_queue.put("FATAL ERROR: CUDA not available. Processing on CPU (will be slow).")
                else:
                    # Acknowledge the memory constraint and the new aggressive setting
                    self.progress_queue.put(f"WARNING: Max quality attempt (PRE_RESIZE_SCALE={PRE_RESIZE_SCALE*100:.0f}%, minimal denoise). Check for 'GPU MEMORY ERROR'.")
                    self.progress_queue.put(f"LOG: PyTorch using {torch.cuda.get_device_name(0)}.")


                # 2. MODEL PRE-LOADING AND SETUP
                model_file = self.model_path.get()
                self.progress_queue.put("LOG: Loading **Large EDSR (32/256)** architecture and applying key mapping...")
                
                model = EDSR_PyTorch_Model(scale=SCALE_FACTOR)
                state_dict = torch.load(model_file, map_location='cpu')
                
                if isinstance(state_dict, dict):
                    if 'model_state_dict' in state_dict:
                         state_dict = state_dict['model_state_dict']
                    elif 'state_dict' in state_dict:
                         state_dict = state_dict['state_dict']
                
                # KEY MAPPING FIX: Rename old keys to new keys 
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('head.0.'):
                        new_key = 'conv_input.' + k[7:]
                        new_state_dict[new_key] = v
                    elif k.startswith('r.'):
                        new_key = 'body.' + k[2:]
                        new_state_dict[new_key] = v
                    elif k.startswith('tail.'):
                        if k.startswith('tail.0.0.'):
                            new_key = 'upsample.0.' + k[9:] 
                            new_state_dict[new_key] = v
                        elif k.startswith('tail.0.2.'):
                            new_key = 'upsample.2.' + k[9:] 
                            new_state_dict[new_key] = v
                        elif k.startswith('tail.1.'):
                            new_key = 'upsample.4.' + k[7:] 
                            new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                # Use strict=False to ignore the MeanShift layer keys.
                model.load_state_dict(new_state_dict, strict=False) 
                
                model.to(DEVICE) 
                model.eval()
                self.progress_queue.put("LOG: **SUCCESS**! EDSR model loaded. Starting image processing loop...")


                # 3. FRAME PROCESSING LOOP
                for input_path, output_path_base in self.job_queue: 
                    if self.stop_flag.is_set():
                        self.progress_queue.put("LOG: Stop signal received.")
                        break

                    try:
                        # 4a. Read and initial conversion
                        img_cv_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
                        if img_cv_bgr is None:
                            self.progress_queue.put(f"FAIL: Could not read image {os.path.basename(input_path)}")
                            continue
                        
                        # --- STAGE 1 DENOISING (MAX DETAIL) ---
                        # Reduces initial sensor noise while preserving maximum detail. (Strength 30, 30)
                        img_cv_bgr_denoised_1 = cv2.fastNlMeansDenoisingColored(img_cv_bgr, None, 30, 30, 7, 21) # Changed from 45 to 30


                        img_rgb = cv2.cvtColor(img_cv_bgr_denoised_1, cv2.COLOR_BGR2RGB)
                        original_h, original_w = img_rgb.shape[:2]
                        
                        # 4b. MEMORY FIX: Resize the image before processing!
                        if PRE_RESIZE_SCALE != 1.0:
                            new_w = int(original_w * PRE_RESIZE_SCALE)
                            new_h = int(original_h * PRE_RESIZE_SCALE)
                            
                            # Downscale using INTER_CUBIC for high quality
                            img_rgb_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            
                            # --- STAGE 2 DENOISING (MINIMAL) ---
                            # Cleans up new compression/scaling artifacts, allowing most detail through. (Strength 2, 2)
                            img_rgb_final_input = cv2.fastNlMeansDenoisingColored(img_rgb_resized, None, 2, 2, 7, 21) # Changed from 4 to 2
                            
                            self.progress_queue.put(f"LOG: Resized {os.path.basename(input_path)} from {original_w}x{original_h} to {new_w}x{new_h}.")
                        else:
                            img_rgb_final_input = img_rgb
                        
                        # 4c. PyTorch Input Preparation
                        img_float_np = img_rgb_final_input.astype(np.float32) 
                        img_normalized_np = img_float_np / 255.0

                        input_tensor = torch.from_numpy(img_normalized_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                        
                        # 4d. Perform Upscaling on GPU
                        enhanced_tensor = model(input_tensor) 

                        # 4e. PyTorch Output Cleanup
                        enhanced_tensor_cpu = enhanced_tensor.squeeze(0).mul(255).clamp(0, 255).cpu()
                        enhanced_img_np_rgb_float = enhanced_tensor_cpu.permute(1, 2, 0).numpy()
                        enhanced_img_np_rgb_uint8 = enhanced_img_np_rgb_float.astype(np.uint8)
                        
                        # 4f. Final BGR conversion
                        final_output_img_bgr = cv2.cvtColor(enhanced_img_np_rgb_uint8, cv2.COLOR_RGB2BGR)

                        
                        # 🌟🌟🌟 ARTIFACT CHECK AND FALLBACK 🌟🌟🌟
                        if self._check_artifacting(final_output_img_bgr):
                            # Artifacts detected. Fall back to clean Bicubic upscale.
                            original_image_bicubic = cv2.imread(input_path, cv2.IMREAD_COLOR)
                            final_output_img_bgr = cv2.resize(
                                original_image_bicubic, 
                                (original_w * SCALE_FACTOR, original_h * SCALE_FACTOR), 
                                interpolation=cv2.INTER_CUBIC
                            )
                            log_status = f"FALLBACK: {os.path.basename(input_path)} used Bicubic (artifacts detected)."
                        else:
                            log_status = f"SUCCESS: {os.path.basename(input_path)} used EDSR."
                        # 🌟🌟🌟 END FALLBACK 🌟🌟🌟


                        # 4g. Save Final Output
                        base_name, _ = os.path.splitext(output_path_base)
                        output_path_png = base_name + '.png'
                        os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
                        
                        cv2.imwrite(output_path_png, final_output_img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]) 

                        self.progress_queue.put(log_status)

                    except Exception as e:
                        if 'out of memory' in str(e).lower():
                            # If this error occurs, they must use a lower scale
                            e_msg = f"GPU MEMORY ERROR: {e}. **Image still too large.** You must lower PRE_RESIZE_SCALE to 0.55 or 0.5."
                        else:
                            e_msg = str(e)
                            
                        self.progress_queue.put(f"FAIL: {os.path.basename(input_path)} - {e_msg}")
                
            except Exception as e:
                self.progress_queue.put(f"Main PyTorch Loop Error: {e}")
            finally:
                self.is_running = False
                self.master.after(0, self._finish_enhancement)

    # --- PROGRESS AND STOPPING (omitted for brevity) ---
    def update_progress(self):
        while True:
            try:
                message = self.progress_queue.get_nowait()
                if message.startswith("SUCCESS") or message.startswith("FALLBACK"):
                    self.processed_count += 1
                    self.log(message)
                elif message.startswith("LOG") or message.startswith("WARNING"):
                    self.log(message)
                elif message.startswith("FATAL ERROR"):
                    self.log(message)
                    self.stop_flag.set()
                    self._finish_enhancement()
                    return
                else:
                    self.log(f"ERROR: {message}")

                if self.total_frames > 0:
                    progress = (self.processed_count / self.total_frames) * 100
                    self.progress_bar['value'] = self.processed_count
                    self.status_var.set(f"Processing... {self.processed_count} of {self.total_frames} frames ({progress:.1f}%)")
            except q_temp.Empty:
                break

        if self.is_running:
            self.master.after(100, self.update_progress)
        elif not self.is_running and not self.stop_flag.is_set() and self.total_frames > 0:
            self.status_var.set(f"Finished! {self.processed_count} frames processed.")
        elif not self.is_running and self.stop_flag.is_set():
             self.status_var.set("Stopped by user.")
        

    def stop_enhancement(self):
        self.stop_flag.set()
        self.is_running = False
        self.status_var.set("Stopping GPU processing. Please wait...")

    def _finish_enhancement(self):
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        if not self.stop_flag.is_set():
            self.status_var.set(f"Finished! {self.processed_count} frames processed.")
        else:
            self.status_var.set("Stopped by user.")


if __name__ == '__main__':
    try:
        # Simple check to ensure necessary libraries are installed for the GUI to launch
        if 'cv2' not in globals():
             print("FATAL ERROR: OpenCV (cv2) library is not imported/installed. Please run 'pip install opencv-python'.")
        if 'torch' not in globals():
             print("FATAL ERROR: PyTorch library is not imported/installed. Please check your PyTorch installation.")

        root = tk.Tk()
        app = BatchImageClarityEnhancer(root)
        root.mainloop()
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")