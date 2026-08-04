import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import glob
import math

class BatchFrameExtractorApp:
    """
    A Tkinter application to extract frames from ALL video files 
    in a chosen source directory, with a real-time progress bar and full-screen mode.
    """
    def __init__(self, master):
        self.master = master
        master.title("Batch Video Frame Extractor")
        
        # Bind the Escape key to the close_window method
        master.bind('<Escape>', self.close_window)
        
        # --- Variables ---
        self.source_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.frame_skip = tk.IntVar(value=1)
        self.is_running = False
        
        # --- Setup GUI ---
        self.setup_gui()

    def close_window(self, event=None):
        """Closes the Tkinter application."""
        if self.is_running:
             # Prevent closing while processing to avoid data corruption
             messagebox.showwarning("Warning", "Please wait for the extraction to finish before closing.")
             return
        self.master.destroy()

    def setup_gui(self):
        """Sets up all GUI elements (labels, buttons, entry fields, and progress bar)."""
        
        # Use a Frame for centering content in full screen
        main_frame = tk.Frame(self.master, padx=50, pady=50)
        main_frame.pack(expand=True, fill='both')

        # 1. Source Directory Selection
        tk.Label(main_frame, text="1. Select Source Directory (Folder containing video files):", font=('Arial', 12)).pack(pady=(10, 0))
        tk.Entry(main_frame, textvariable=self.source_dir, width=100, state='readonly').pack(padx=10)
        tk.Button(main_frame, text="Browse Source Folder", command=self.select_source_dir).pack()
        
        # 2. Output Directory Selection
        tk.Label(main_frame, text="2. Select Output Directory (Folder to save all extracted frames):", font=('Arial', 12)).pack(pady=(20, 0))
        tk.Entry(main_frame, textvariable=self.output_dir, width=100, state='readonly').pack(padx=10)
        tk.Button(main_frame, text="Browse Destination Folder", command=self.select_output_dir).pack()
        
        # 3. Frame Skip/Interval (Optional)
        tk.Label(main_frame, text="3. Frame Interval (1 for every frame, 10 for every 10th frame):", font=('Arial', 12)).pack(pady=(20, 0))
        tk.Spinbox(main_frame, from_=1, to=100, textvariable=self.frame_skip, width=5, font=('Arial', 10)).pack()

        # 4. Process Button
        self.start_button = tk.Button(main_frame, text="Start Batch Extraction", command=self.start_batch_extraction, bg="darkorange", fg="white", font=('Arial', 14, 'bold'))
        self.start_button.pack(pady=(30, 20))
        
        # --- Progress Bar and Status ---
        # Container for progress bar and status to manage layout at the bottom
        progress_container = tk.Frame(self.master)
        progress_container.pack(side=tk.BOTTOM, fill=tk.X, padx=50, pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(progress_container, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 5))

        self.status_label = tk.Label(progress_container, text="Ready | Press ESC to Exit", relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=5)

        # 5. Exit Button
        tk.Button(main_frame, text="Exit / Close", command=self.close_window, bg="red", fg="white").pack(pady=20, side=tk.BOTTOM)


    def select_source_dir(self):
        path = filedialog.askdirectory(title="Choose the folder containing video files")
        if path:
            self.source_dir.set(path)
            
    def select_output_dir(self):
        path = filedialog.askdirectory(title="Choose a folder to save ALL the frames")
        if path:
            self.output_dir.set(path)

    def start_batch_extraction(self):
        """Validates inputs, finds video files, and starts the batch process."""
        if self.is_running:
            messagebox.showwarning("Busy", "Extraction is already in progress.")
            return

        source_folder = self.source_dir.get()
        output_folder = self.output_dir.get()
        frame_interval = self.frame_skip.get()

        if not all([os.path.isdir(source_folder), os.path.isdir(output_folder), frame_interval >= 1]):
            messagebox.showerror("Error", "Please check your folder selections and frame interval.")
            return

        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        all_videos = []
        for ext in video_extensions:
            all_videos.extend(glob.glob(os.path.join(source_folder, ext)))
        
        if not all_videos:
            messagebox.showinfo("No Videos Found", f"No supported video files found in:\n{source_folder}")
            return
        
        # Disable button and set running state
        self.is_running = True
        self.start_button.config(state=tk.DISABLED, text="Processing...")
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = len(all_videos)

        total_videos = len(all_videos)
        total_frames_extracted = 0
        
        try:
            for i, video_file in enumerate(all_videos):
                video_name = os.path.basename(video_file)
                self.status_label.config(text=f"Processing Video {i+1} of {total_videos}: {video_name}")
                self.master.update_idletasks() # Force GUI update

                extracted_count = self.process_single_video_with_progress(video_file, output_folder, frame_interval)
                total_frames_extracted += extracted_count
                
                self.progress_bar['value'] = i + 1
                self.master.update_idletasks()

            self.status_label.config(text="Batch Extraction Complete.")
            messagebox.showinfo(
                "Batch Complete", 
                f"Finished processing {total_videos} videos.\nTotal frames extracted: {total_frames_extracted}."
            )

        except Exception as e:
            self.status_label.config(text=f"An error occurred: {e}")
            messagebox.showerror("Extraction Failed", f"An unexpected error occurred: {e}")
            
        finally:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL, text="Start Batch Extraction")
            self.progress_bar['value'] = 0
            self.status_label.config(text="Ready | Press ESC to Exit")
            cv2.destroyAllWindows()


    def process_single_video_with_progress(self, video_file, output_folder, interval):
        """
        Handles video reading and frame saving with frame-by-frame progress updates.
        """
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Skipping file: Could not open video {video_file}")
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_saved_frames = math.ceil(total_frames / interval) if interval > 0 else 0
        
        frame_count = 0
        extracted_count = 0
        
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        save_path = os.path.join(output_folder, f"{video_name}_frames")
        os.makedirs(save_path, exist_ok=True)
        
        # Use the main progress bar for fine-grained progress within the video
        # We need to temporarily change its range to reflect the current video's expected frames
        original_max = self.progress_bar['maximum']
        original_value = self.progress_bar['value']
        
        self.progress_bar['maximum'] = expected_saved_frames if expected_saved_frames > 0 else total_frames
        self.progress_bar['value'] = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % interval == 0:
                image_filename = os.path.join(save_path, f"{video_name}_frame_{extracted_count:05d}.jpg")
                cv2.imwrite(image_filename, frame)
                extracted_count += 1
                
                # Update the main progress bar
                self.progress_bar['value'] = extracted_count
                self.master.update_idletasks() # Crucial for real-time update
            
            frame_count += 1

        cap.release()
        
        # Restore the main progress bar's range and value to reflect batch progress
        self.progress_bar['maximum'] = original_max
        self.progress_bar['value'] = original_value + 1
        
        return extracted_count


if __name__ == "__main__":
    root = tk.Tk()
    # 💥 FULL-SCREEN COMMAND 💥
    root.attributes('-fullscreen', True)
    
    app = BatchFrameExtractorApp(root)
    root.mainloop()