import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os

class JSONValidatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JSON Validator")

        self.load_button = tk.Button(root, text="Load JSON", command=self.load_json)
        self.load_button.pack()

        self.accept_button = tk.Button(root, text="Accept", command=self.accept_entry, state=tk.DISABLED)
        self.accept_button.pack(side=tk.LEFT, padx=10)

        self.reject_button = tk.Button(root, text="Reject", command=self.reject_entry, state=tk.DISABLED)
        self.reject_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(root, text="Save Accepted", command=self.save_accepted, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.save_progress_button = tk.Button(root, text="Save Progress", command=self.save_progress, state=tk.DISABLED)
        self.save_progress_button.pack(side=tk.LEFT, padx=10)

        self.display_area = tk.Text(root, height=20, width=80)
        self.display_area.pack(pady=10)

        self.json_data = []
        self.current_index = 0
        self.accepted_entries = []

        # Bind arrow keys to functions
        root.bind('<Right>', lambda event: self.accept_entry())
        root.bind('<Left>', lambda event: self.reject_entry())

    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as file:
                self.json_data = json.load(file)
                self.current_index = 0
                self.accepted_entries = []
                self.show_current_entry()
                self.accept_button.config(state=tk.NORMAL)
                self.reject_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.DISABLED)
                self.save_progress_button.config(state=tk.NORMAL)
                self.load_progress()

    def show_current_entry(self):
        if self.current_index < len(self.json_data):
            entry = self.json_data[self.current_index]
            self.display_area.delete(1.0, tk.END)
            self.display_area.insert(tk.END, json.dumps(entry, indent=4))
        else:
            self.display_area.delete(1.0, tk.END)
            self.display_area.insert(tk.END, "No more entries to review.")

    def accept_entry(self):
        if self.current_index < len(self.json_data):
            self.accepted_entries.append(self.json_data[self.current_index])
            self.current_index += 1
            self.show_current_entry()
            if self.current_index == len(self.json_data):
                self.save_button.config(state=tk.NORMAL)

    def reject_entry(self):
        if self.current_index < len(self.json_data):
            self.current_index += 1
            self.show_current_entry()
            if self.current_index == len(self.json_data):
                self.save_button.config(state=tk.NORMAL)

    def save_accepted(self):
        if self.accepted_entries:
            file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if file_path:
                with open(file_path, 'w') as file:
                    json.dump(self.accepted_entries, file, indent=4)
                messagebox.showinfo("Saved", f"Accepted entries saved to {file_path}")

    def save_progress(self):
        progress_data = {
            "current_index": self.current_index,
            "accepted_entries": self.accepted_entries
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(progress_data, file, indent=4)
            messagebox.showinfo("Saved", f"Progress saved to {file_path}")

    def load_progress(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], title="Load Progress")
        if file_path:
            with open(file_path, 'r') as file:
                progress_data = json.load(file)
                self.current_index = progress_data.get("current_index", 0)
                self.accepted_entries = progress_data.get("accepted_entries", [])
                self.show_current_entry()
                messagebox.showinfo("Progress Loaded", f"Progress loaded from {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = JSONValidatorApp(root)
    root.mainloop()
