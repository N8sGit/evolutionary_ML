# evolutionary_machine_learning
Machine learning experiment that uses evolutionary algorithms instead of traditional methods for model selection.

Initial results have proven promising, but more work will have to go into optimization and performance to actually put it ahead as a viable attractive replacement for traditional machine learning development methods. 

### What is neuro-evolution? 

Neuro-evolution is a heterodox approach to machine learning and artificial neural network development which incorporates mechanisms from genetic programming. The field has been around at least since the early 1990s, but really came into its own following K. Stanley and R. Miikkulainen's work on NeuroEvolution of Augmenting Topologies [(NEAT)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

One of the most appealing features of these methods is that it allows one to explore all facets of the relevant search space at once. One can mutate hyperparameters, layers, and activation functions simultaneously, which combines many of the different efforts one has to perform individually into one. 

The upfront time and resource expense is large, but the argument can be made that once you factor in how much work the algorithm is actually doing for you it pays for itself eventually. 

The magic of evolutionary discovery algorithms is that all you must do is hand it the right set of primtive combinitorial components (architecture elements), give it a general idea of what differentiates better from worse, and set it loose as it then explores the space of architectural possibilities until something sticks. 

I'm nearly convinced if we could land on the right recipe we wouldn't do R&D any other way. The main challenge seems be selecting the right assumptions about what areas of model space ought to be explored and making sure the right fitness function and parameters have been chosen. We can't evolve the code we wrote alongside the code we generate.

Initially experiments have shown that the algorithm will consistently outperform a regular model for the MNIST dataset if given enough time, and will do the same for an unoptimized base model for CIFAR10 (although much needs to be done before it can take on such a large and complex dataset comfortably). 

Here I just wanted to share my initial findings and the codebase in their current, rough shape. I may go back to the drawingboard anew and reapply my learnings afresh. 

Currently, the program works reasonably well for MNIST but will tank on CIFAR10. In retrospect I flew a little close to the sun with the latter and will need to revisit it. You need to get into the weeds with how you allocate your resources if you want this algorithm to scale. 

This project is just my first attempts at educating myself about neuro-evolution and shouldn't be taken that seriously.

### Running the project
Assuming you have all the regular python tools for machine learning development setup, in the root directory in the terminal run:  
```pip install requirements.txt```

To run the main loop : ```python main.py```

You can set the flag for which dataset you want it to work on in ```data.py```. However it's recommended to keep it to MNIST for the time being.  
