"""
possible input should be:
a.  hl = Nr of hidden layers
b.  haf = Hidden activation function
c.  oa = Output activation
d.  cf = Cost function
e.  lr = Learning rate
f.  Initial weight range
g.  Data source – one of:
    i.  File 
    ii. Function (see tflowtools.py)
h.  Case fraction
i.  Validation fraction (VaF = [0,1])
    i.  S*VaF
j.  Test fraction (TeF = [0,1])
    i.  S*TeF
k.  Training fraction (1-(VaF+TeF))
    i.  S*(1 –(TeF+VaF))
l.  Minibatch size
m.  Map batch size
n.  Steps
o.  Map Layers
p.  Map dendrograms
q.  Display weights
r.  Display biases
s.  Vint(validation interval)
t.  UNSURE about these: 
    i.  Nodes in each layer
    ii. Error function
    iii.  Initial weight values
"""

import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.run_ANN = tk.Button(self)
        self.run_ANN["text"] = "Run ANN"
        self.run_ANN["command"] = self.run
        self.run_ANN.pack(side="left")

        self.init_ANN = tk.Button(self)
        self.init_ANN["text"] = "Initialize ANN"
        self.init_ANN["command"] = self.initialize
        self.init_ANN.pack(side="right")

        self.quit = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="bottom")

    def initialize_network(self):
        print("initializing network with following input: ")
        for argument in arguments:
          print(argument[0])
          print(argument[1])
    def run_network(self):
        print("Running network with following input: ")
        for i, argument in enumerate(self.arguments):
          print(i ".  "argument[0] + ": " argument[1])

root = tk.Tk()
app = Application(master=root)
app.mainloop()

