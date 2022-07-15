from Q_learning_interface import QLearningInterface

model_parameters = {
    "states":  ['up_24', 'down_24', 'up_27', 'down_27'], 
    'actions': [24, 27],
    "epsilon": 1,
    "learning_rate": 0.5,
    "discount_factor": 0.1
}
q_lean = QLearningInterface(model_parameters=None, model_path='models/', model_name = 'bciAgent')
q_lean.save_model()