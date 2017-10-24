
import sys
sys.path.append('./')
sys.path.append('./tools')
from tkinter import *
from tkinter import ttk
from gann import Gann, Caseman
import tflowTools as TFT


class Application():
    def __init__(self):

        root = Tk()

        root.title("General Artificial Neural Network Interface")
        mainframe = ttk.Frame(root, padding="20 20 20 20")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        self.valid_input = False

        self.data_sources = [("Parity", 0),
                             ("Auto Encoder", 1),
                             ("Bit Counter", 2),
                             ("Segment Counter", 3),
                             ("MNIST", 4),
                             ("Wine Quality", 5),
                             ("Glass", 6),
                             ("Yeast", 7),
                             ("Hacker's Choice", 8)]

        self.data_source = IntVar()
        self.data_source.set(0)
        self.parameters = StringVar()

        self.labels = ["Network Dimensions",
                       "Hidden Activation Function",
                       "Output Activation Function",
                       "Cost Function",
                       "Learning Rate",
                       "Initial Weight Range",
                       "Epochs",
                       "Validation Fraction",
                       "Test Fraction",
                       "Minibatch Size",
                       "Map Batch Size",
                       "Validation intervall",
                       "Steps",
                       "Map Layers",
                       "Map Dendrograms",
                       "Display Weights",
                       "Display Biases", ]

        self.label_details = ["Eg. '2,25,12,2'",
                              "See tensorflow documentation for list of supported functions",
                              "See tensorflow documentation for list of supported functions",
                              "'cross entropy' or 'mse'",
                              "",
                              "Eg. '0.3-0.7'",
                              "",
                              "",
                              "",
                              "",
                              "'0' = No map test",
                              "",
                              "",
                              "Eg. '2-3'",
                              "Eg. '1,3,4'",
                              "",
                              "", ]

        self.network_settings = []
        self.dims = []
        self.hidden_activation_function = ""
        self.output_activation_function = ""
        self.loss_function = ""
        self.learning_rate = 0
        self.initial_weight_range = []
        self.epochs = 0
        self.validationFraction = 0
        self.testFraction = 0
        self.minibatch_size = 0
        self.map_batch_size = 0
        self.validation_interval = 0
        self.steps = 0
        self.map_layers = 0
        self.map_dendrograms = 0
        self.display_weights = 0
        self.display_biases = 0

        for i in range(len(self.labels)):
            self.network_settings.append(StringVar())

        self.create_widgets(mainframe)

        root.mainloop()

    def create_widgets(self, frame):

        ttk.Label(frame, text="Data Source:", font=("Arial", 20)).grid(
            column=0, pady=10, padx=10)

        for (lbl, val) in self.data_sources:
            ttk.Radiobutton(frame, text=lbl,
                            variable=self.data_source, value=val).grid(column=0, padx=15, pady=5)

        ttk.Label(frame, text="Parameters:", font=("Arial", 10, "italic")).grid(
            column=0)

        ttk.Entry(frame, textvariable=self.parameters).grid(
            column=0)

        ttk.Label(frame, text="Settings:", font=("Arial", 20)).grid(
            columnspan=3, row=0, column=1, pady=10)

        for i in range(0, len(self.labels)):
            ttk.Label(frame, text=self.labels[i]).grid(
                column=1, row=i + 1, sticky=E, padx=5, pady=5)
            ttk.Entry(frame, text="", textvariable=self.network_settings[i]).grid(
                column=2, row=i + 1, sticky=E, padx=5, pady=5)
            ttk.Label(frame, text=self.label_details[i], font=("Arial", 10, "italic")).grid(
                column=3, row=i + 1, sticky=W, padx=5, pady=5)

        ttk.Button(frame, text="Validate",
                   command=self.validate).grid(column=0, row=20)
        ttk.Button(frame, text="Initialize",
                   command=self.initialize).grid(column=1, row=20)

    def validate(self):
        try:
            self.dims = [int(x)
                         for x in self.network_settings[0].get().split(",")]

            self.hidden_activation_function = self.network_settings[1].get()
            self.output_activation_function = self.network_settings[2].get()
            self.loss_function = self.network_settings[3].get()
            self.learning_rate = float(self.network_settings[4].get())
            self.initial_weight_range = [
                float(x) for x in self.network_settings[5].get().split("-")]
            self.epochs = int(self.network_settings[6].get())
            self.validation_fraction = float(self.network_settings[7].get())
            self.test_fraction = float(self.network_settings[8].get())
            self.minibatch_size = int(self.network_settings[9].get())
            self.map_batch_size = int(self.network_settings[10].get())
            self.validation_interval = int(self.network_settings[11].get())
            self.steps = int(self.network_settings[12].get())
            self.map_layers = int(self.network_settings[13].get())
            self.dendrograms = int(self.network_settings[14].get())
            self.display_weights = int(self.network_settings[15].get())
            self.biases = int(self.network_settings[16].get())
            self.valid_input = True

        except Exception as e:
            print(str(e))
            self.valid_input = False

    def initialize(self):
        case_generator = None

        epochs = 100
        lrate = 0.1
        showint = 0
        mbs = None
        vfrac = 0.1
        tfrac = 0.1
        vint = 0
        sm = True

        bestk = 1

        size = 10
        mbs = mbs if mbs else size
        case_generator = (lambda: TFT.gen_all_parity_cases(10))
        cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
        ann = Gann(dims=[10, 8, 2], cman=cman, lrate=lrate,
                   showint=showint, mbs=mbs, vint=vint, softmax=sm)
        # Plot a histogram and avg of the incoming weights to module 0.
        ann.gen_probe(0, 'wgt', ('hist', 'avg'))
        # Plot average and max value of module 1's output vector
        # ann.gen_probe(1, 'out', ('avg', 'max'))
        # Add a grabvar (to be displayed in its own matplotlib window).
        # ann.add_grabvar(0, 'wgt')
        ann.run(epochs, bestk=bestk)
        return ann

    def initialize_network(self, epochs=50, nbits=10, ncases=500, lrate=0.5, showint=500, mbs=20, vfrac=0.1, tfrac=0.1, vint=200, sm=True, bestk=1):
        print("initializing network with following inputs: ")
        print(self.entry.get())

        for i, argument in enumerate(self.arguments):
            print(str(i + 1) + ".  " + argument[0] + ": " + argument[1])
        case_generator = (lambda: TFT.gen_vector_count_cases(ncases, nbits))
        cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
        self.ann = Gann(dims=[nbits, nbits * 3, nbits + 1], cman=cman,
                        lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)

    def run_network(self, epochs=50, bestk=None):
        print("Running network with following input: ")
        for i, argument in enumerate(self.arguments):
            print(str(i + 1) + ".  " + argument[0] + ": " + argument[1])
        self.ann.run(epochs, bestk=bestk)


app = Application()
app.mainloop()
