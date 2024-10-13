import torch
from evolution.algorithm import evolutionary_algorithm
from training import train_standard_model
from comparison import compare_models

if __name__ == "__main__":
    # Run the evolutionary algorithm
    best_model, best_fitness = evolutionary_algorithm(pop_size=10, generations=3, base_mutation_rate=0.1)
    print(f"Best model found by evolutionary algorithm has fitness: {best_fitness}")
    
    # Save the best model from evolutionary algorithm
    torch.save(best_model.state_dict(), 'best_model_evolution.pth')
    
    # Train a standard model for comparison
    trained_model = train_standard_model()
    
    # Save the standard trained model
    torch.save(trained_model['model'].state_dict(), 'standard_model.pth')
    
    # Compare both models across evaluations
    compare_models(best_model, trained_model['model'])