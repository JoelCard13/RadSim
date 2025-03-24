from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gc

YRS = 31557600   # Number of Seconds in a 365.25-Day Year
DAYS = 86400     # Number of Seconds in a 24-Hour Day

ISOTOPES = [
    {"name": "Carbon-14", "half_life": 5730*YRS, "molar_mass":14},{"name": "Uranium-238", "half_life": 4.468e9*YRS, "molar_mass":238},
    {"name": "Iodine-131", "half_life": 8.04*DAYS, "molar_mass":131},{"name": "Tritium", "half_life": 12.35*YRS, "molar_mass":3},
    {"name": "Radon-222", "half_life": 3.8*DAYS, "molar_mass": 222},{"name": "Plutonium-239", "half_life": 24065*YRS, "molar_mass": 239},
    {"name": "Americium-241", "half_life": 432.2*YRS, "molar_mass":241},{"name": "Lutetium-177", "half_life": 6.71*DAYS, "molar_mass":177},
    {"name": "Barium-133", "half_life": 10.74*YRS, "molar_mass":133},{"name": "Molybdenum-99", "half_life": 66*3600, "molar_mass":99},
    {"name": "Bismuth-212", "half_life": 60.55*60, "molar_mass":212},{"name": "Nickel-63", "half_life": 96*YRS, "molar_mass":63},
    {"name": "Cadmium-109", "half_life": 464*DAYS, "molar_mass":109},{"name": "Phosphorus-32", "half_life": 14.29*DAYS, "molar_mass":32},
    {"name": "Calcium-45", "half_life": 163*DAYS, "molar_mass":45},{"name": "Phosphorus-33", "half_life": 25.4*DAYS, "molar_mass":33},
    {"name": "Cesium-137", "half_life": 30*YRS, "molar_mass":137},{"name": "Polonium-210", "half_life": 138.3*DAYS, "molar_mass":210},
    {"name": "Chlorine-36", "half_life": 301000*YRS, "molar_mass":36},{"name": "Radium-226", "half_life": 1600*YRS, "molar_mass":226},
    {"name": "Chromium-51", "half_life": 27.704*DAYS, "molar_mass":51},{"name": "Cobalt-57", "half_life": 271*DAYS, "molar_mass":57},
    {"name": "Rhenium-188", "half_life": 16.98*3600, "molar_mass":188},{"name": "Cobalt-58", "half_life": 70.8*DAYS, "molar_mass":58},
    {"name": "Cobalt-58", "half_life": 70.8*DAYS, "molar_mass":58},{"name": "Rubidium-81", "half_life": 4.58*3600, "molar_mass":81},
    {"name": "Cobalt-60", "half_life": 5.27*YRS, "molar_mass":60},{"name": "Selenium-75", "half_life": 119.8*DAYS, "molar_mass":75},
    {"name": "Copper-62", "half_life": 9.74*60, "molar_mass":62},{"name": "Sodium-22", "half_life": 2.6*YRS, "molar_mass": 22},
    {"name": "Copper-64", "half_life": 12.7*3600, "molar_mass":64},{"name": "Sodium-24", "half_life": 15*3600, "molar_mass":24},
    {"name": "Copper-67", "half_life": 61.86*3600, "molar_mass":67},{"name": "Strontium-85", "half_life": 64.84*DAYS, "molar_mass":85},
    {"name": "Gallium-67 ", "half_life": 78.26*3600, "molar_mass":67},{"name": "Strontium-89", "half_life": 50.5*DAYS, "molar_mass":89},
    {"name": "Gallium-68", "half_life": 68*60, "molar_mass":68},{"name": "Strontium-90", "half_life": 29.12*YRS, "molar_mass":90},
    {"name": "Gold-195", "half_life": 183*DAYS, "molar_mass":195},{"name": "Sulfur-35", "half_life": 87.44*DAYS, "molar_mass":35},
    {"name": "Technetium-99", "half_life": 213000*YRS, "molar_mass":99},{"name": "Indium-111", "half_life": 2.83*DAYS, "molar_mass":111},
    {"name": "Technetium-99m", "half_life": 6.02*3600, "molar_mass":99},{"name": "Indium-113m", "half_life": 1.658*3600, "molar_mass":113},
    {"name": "Tin-113", "half_life": 115.1*DAYS, "molar_mass":113},{"name": "Iodine-123", "half_life": 13.2*3600, "molar_mass":123},
    {"name": "Tungsten-188", "half_life": 69.4*DAYS, "molar_mass":188},{"name": "Iodine-125", "half_life": 60.14*DAYS, "molar_mass":125},
    {"name": "Uranium-235", "half_life": 704e6*YRS, "molar_mass":235},{"name": "Iodine-129", "half_life": 157e5*YRS, "molar_mass":129},
    {"name": "Xenon-127", "half_life": 36.41*DAYS, "molar_mass":127},{"name": "Iron-55", "half_life": 2.7*YRS, "molar_mass":55},
    {"name": "Xenon-133", "half_life": 5.245*DAYS, "molar_mass":133},{"name": "Iron-59", "half_life": 44.529*DAYS, "molar_mass":59},
    {"name": "Yttrium-90", "half_life": 64*3600, "molar_mass":90},{"name": "Krypton-81m", "half_life": 13, "molar_mass":81},
    {"name": "Ytterbium-169", "half_life": 32.01*DAYS, "molar_mass":169},{"name": "Krypton-85", "half_life": 10.72*YRS, "molar_mass":85}
]

AVOGADRO = 6.022e23

class RadSim:
    def __init__(self):
        self.root = tk.Tk()
        self.root.iconphoto(True, tk.PhotoImage(file='greenatom.png'))
        self.root.title("Radioactive Decay Simulator")
        self.root.geometry("525x540")
        self.root.config(bg="gray12")
        self.root.resizable(False, False)
        big_font = tkFont.Font(family="Arial", size=20, weight=tkFont.BOLD)
        lil_font = tkFont.Font(family="Arial", size=18, weight=tkFont.BOLD)
        text_font = tkFont.Font(family="Arial", size=18)
        tiny_font = tkFont.Font(family="Arial", size=12, weight=tkFont.BOLD)
        
        self.ani = None
        self.paused = False

        tk.Label(self.root, text="WELCOME TO RADSIM", font=big_font, bg="gray12",
                 fg="greenyellow").pack(padx=10, pady=20)
        tk.Label(self.root, text="ENTER ISOTOPE AND WEIGHT IN GRAMS",
                 font=lil_font, bg="gray12", fg="greenyellow").pack(padx=10)

        entryframe = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, bg="gray12")
        entryframe.columnconfigure(0, weight=1)
        entryframe.columnconfigure(1, weight=1)

        self.isoentry = tk.Entry(entryframe, font=text_font)
        self.isoentry.grid(row=0, column=0, sticky=tk.W+tk.E)
        self.isoentry.insert(0,"Carbon-14")
        self.isoentry.config(fg="white", bg="gray30")
        self.massentry = tk.Entry(entryframe, font=text_font)
        self.massentry.grid(row=0, column=1, sticky=tk.W+tk.E)
        self.massentry.insert(0,"1")
        self.massentry.config(fg="white", bg="gray30")
        entryframe.pack(padx=10, pady=20, fill='x')

        tk.Label(self.root, text="NOTE: USE A LOWER WEIGHT FOR SHORTER HALF LIVES FOR BETTER RESULTS",
                 font=tiny_font, bg="gray12", fg="yellowgreen").pack(padx=10, pady=10)

        tk.Button(self.root, text="RUN SIMULATION", font=big_font,
                  command=self.run_simulation).pack(padx=10, pady=10)
        
        self.btn_pause = tk.Button(self.root, text="Pause", state=tk.DISABLED,
                                   command=self.toggle_pause, font=('Arial', 14))
        self.btn_pause.pack(pady=10)
        
        self.load_gif("decay.gif")
        self.root.mainloop()

    def load_gif(self, gif_path, width=150, height=200):
        try:
            self.gif_frames = []
            with Image.open(gif_path) as img:
                while True:
                    resized_frame = img.copy().resize((width, height), resample=Image.LANCZOS)
                    frame = ImageTk.PhotoImage(resized_frame)
                    self.gif_frames.append(frame)
                    try:
                        img.seek(img.tell() + 1)
                    except EOFError:
                        break
        
            self.gif_label = tk.Label(self.root, bg="greenyellow")
            self.gif_label.pack(side="bottom", pady=10)
            self.animate_gif(0)
        
        except FileNotFoundError:
            messagebox.showwarning("GIF Error", "Animation file not found!")

    def animate_gif(self, frame_num):
        frame = self.gif_frames[frame_num]
        self.gif_label.config(image=frame)
        next_frame = (frame_num + 1) % len(self.gif_frames)
        self.root.after(50, lambda: self.animate_gif(next_frame))

    def toggle_pause(self, event=None): 
        if self.ani:
            self.paused = not self.paused
            if self.paused:
                self.ani.event_source.stop()
                self.btn_pause.config(text="Resume")
                if hasattr(self, 'mpl_btn_pause'):
                    self.mpl_btn_pause.label.set_text("Resume")
            else:
                self.ani.event_source.start()
                self.btn_pause.config(text="Pause")
                if hasattr(self, 'mpl_btn_pause'):
                    self.mpl_btn_pause.label.set_text("Pause")
            self.ani._fig.canvas.draw_idle()

    def run_simulation(self):
        try:
            iso_name = self.isoentry.get()
            isotope = next(iso for iso in ISOTOPES if iso["name"] == iso_name)
            mass = float(self.massentry.get())
            if mass <= 0:
                raise ValueError("Mass must be positive")            
        except StopIteration:
                messagebox.showerror("Isotope Not Found", "Sorry, that isotope isn't cool enough")
                return        
        except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
                return
        
        atoms = int(mass/isotope["molar_mass"] * AVOGADRO)
        prob = np.log(2)/isotope["half_life"]

        self.btn_pause.config(state=tk.NORMAL)
        self.btn_pause.config(text="Pause")
        self.paused = False

        self.simulate(atoms, prob)

    def simulate(self, atoms, prob):
        if self.ani:
            self.ani.event_source.stop()
            plt.close(self.ani._fig)

        STEPS_PER_FRAME = 50
        MAX_FRAMES = 100
        MAX_ATOMS_FOR_BINOMIAL = 1e18
        MAX_LAMBDA_POISSON = 1e9
        remaining_atoms = atoms
        decay_counts = []
        fig, ax = plt.subplots()

        ax_pause = plt.axes([0.015, 0.015, 0.1, 0.05])
        self.mpl_btn_pause = plt.Button(ax_pause, 'Pause', color='#caea89')
        self.mpl_btn_pause.on_clicked(self.toggle_pause)

        def init():
            ax.clear()
            return ax,

        def update(frame):
            nonlocal remaining_atoms, decay_counts

            if self.paused:
                return ax,
            
            for _ in range(STEPS_PER_FRAME):
                if remaining_atoms <= 0:
                    break

                lambda_decay = remaining_atoms * prob

                if lambda_decay > MAX_LAMBDA_POISSON:
                    mu = lambda_decay
                    sigma = np.sqrt(mu)
                    decays = np.random.normal(mu, sigma)
                    decays = int(np.round(max(0, min(decays, remaining_atoms))))
                elif remaining_atoms > MAX_ATOMS_FOR_BINOMIAL:
                    decays = np.random.poisson(lambda_decay)
                else:
                    decays = np.random.binomial(int(remaining_atoms), prob)
                    decays = min(decays, remaining_atoms)

                remaining_atoms -= decays
                decay_counts.append(decays)

                if len(decay_counts) % 1000 == 0:
                    gc.collect()

            ax.clear()
            ax.hist(decay_counts, bins=30, density=True, alpha=0.6,
                     color='yellowgreen', label='Observed Decays')
            if len(decay_counts) > 1:
                mean = np.mean(decay_counts)
                std = np.std(decay_counts)
                if std > 0:
                    x = np.linspace(mean - 4*std, mean + 4*std, 100)
                    y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mean)/std)**2)
                    ax.plot(x, y, color='darkgreen', linewidth=2, label='Normal Fit')

            ax.set_xlabel('Decay Rate (Decays/Sec)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Decay Distribution (Total Decays: {len(decay_counts)})')
            ax.legend()
            return ax,

        self.ani = FuncAnimation(fig, update, frames=None, init_func=init,
                                 save_count=MAX_FRAMES, blit=False, cache_frame_data=False,
                                 interval=100, repeat=False)
        
        def on_close(event):
            self.btn_pause.config(state=tk.DISABLED)
            self.ani = None
        fig.canvas.mpl_connect('close_event', on_close)
        
        plt.show()

    def on_simulation_end(self):
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_pause.config(text="Pause")
        self.paused = False

if __name__ == "__main__":
    RadSim()