from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import random
import torch.nn as nn
import os
import platform

app = Flask(__name__)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Global variables to maintain game state
game_state = {
    'width': 400,
    'height': 400,
    'paddle_width': 10,
    'paddle_height': 60,
    'ball_size': 10,
    'paddle_speed': 5,
    'ball_speed': 2.5,  # Reduced from 5 to 2.5 to slow down the ball
    'human_paddle_pos': 200,  # Start in the middle
    'ai_paddle_pos': 200,     # Start in the middle
    'ball_pos': [200, 200],   # Start in the middle
    'ball_direction': [1, 1], # Initial direction
    'human_score': 0,
    'ai_score': 0,
    'game_active': True
}

# Load the AI model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = DQN(5, 3).to(device)

# Check if model file exists before loading
model_path = 'static/models/pong_model.pth'
if os.path.exists(model_path):
    ai_model.load_state_dict(torch.load(model_path, map_location=device))
    ai_model.eval()
else:
    print(f"Warning: Model file {model_path} not found!")

def get_ai_state():
    # Convert game state to AI input format
    return np.array([
        game_state['ai_paddle_pos'] / game_state['height'],
        game_state['ball_pos'][1] / game_state['height'],
        (game_state['width'] - game_state['ball_pos'][0]) / game_state['width'],  # Flip x-coordinate for AI
        -game_state['ball_direction'][0],  # Flip x-direction for AI
        game_state['ball_direction'][1]
    ])

def normalize_ball_direction():
    # Normalize ball direction vector
    direction = game_state['ball_direction']
    length = np.sqrt(direction[0]**2 + direction[1]**2)
    game_state['ball_direction'] = [direction[0]/length, direction[1]/length]

def reset_ball(direction):
    game_state['ball_pos'] = [game_state['width'] // 2, game_state['height'] // 2]
    game_state['ball_direction'] = [direction, random.uniform(-1, 1)]
    normalize_ball_direction()

def update_game_state(human_move):
    # Update human paddle position based on command type:
    # If human_move is a string ("up" or "down") then update incrementally,
    # otherwise if it is a number, set the paddle position to that value.
    if isinstance(human_move, str):
        if human_move == 'up':
            game_state['human_paddle_pos'] = max(
                game_state['paddle_height'] // 2,
                game_state['human_paddle_pos'] - game_state['paddle_speed']
            )
        elif human_move == 'down':
            game_state['human_paddle_pos'] = min(
                game_state['height'] - game_state['paddle_height'] // 2,
                game_state['human_paddle_pos'] + game_state['paddle_speed']
            )
    elif isinstance(human_move, (int, float)):
        # Set position directly with clamping
        game_state['human_paddle_pos'] = max(
            game_state['paddle_height'] // 2,
            min(int(human_move), game_state['height'] - game_state['paddle_height'] // 2)
        )
    
    # Update AI paddle position
    state = torch.FloatTensor(get_ai_state()).unsqueeze(0).to(device)
    with torch.no_grad():
        action = ai_model(state).max(1)[1].item()
    
    if action == 1:  # Move up
        game_state['ai_paddle_pos'] = max(
            game_state['paddle_height'] // 2,
            game_state['ai_paddle_pos'] - game_state['paddle_speed']
        )
    elif action == 2:  # Move down
        game_state['ai_paddle_pos'] = min(
            game_state['height'] - game_state['paddle_height'] // 2,
            game_state['ai_paddle_pos'] + game_state['paddle_speed']
        )
    
    # Update ball position
    game_state['ball_pos'][0] += game_state['ball_speed'] * game_state['ball_direction'][0]
    game_state['ball_pos'][1] += game_state['ball_speed'] * game_state['ball_direction'][1]
    
    # Ball collision with top and bottom
    if game_state['ball_pos'][1] <= 0 or game_state['ball_pos'][1] >= game_state['height']:
        game_state['ball_direction'][1] *= -1
    
    # Ball collision with paddles
    # Human paddle
    if (game_state['ball_pos'][0] <= game_state['paddle_width'] + game_state['ball_size']//2 and
        abs(game_state['ball_pos'][1] - game_state['human_paddle_pos']) < game_state['paddle_height'] // 2):
        game_state['ball_direction'][0] *= -1
        # Add randomness to bounce
        game_state['ball_direction'][1] += random.uniform(-0.2, 0.2)
        normalize_ball_direction()
    
    # AI paddle
    if (game_state['ball_pos'][0] >= game_state['width'] - game_state['paddle_width'] - game_state['ball_size']//2 and
        abs(game_state['ball_pos'][1] - game_state['ai_paddle_pos']) < game_state['paddle_height'] // 2):
        game_state['ball_direction'][0] *= -1
        game_state['ball_direction'][1] += random.uniform(-0.2, 0.2)
        normalize_ball_direction()
    
    # Score points
    if game_state['ball_pos'][0] <= 0:
        game_state['ai_score'] += 1
        reset_ball(1)  # Ball moves towards human
    elif game_state['ball_pos'][0] >= game_state['width']:
        game_state['human_score'] += 1
        reset_ball(-1)  # Ball moves towards AI

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset_game():
    game_state['human_paddle_pos'] = game_state['height'] // 2
    game_state['ai_paddle_pos'] = game_state['height'] // 2
    game_state['ball_pos'] = [game_state['width'] // 2, game_state['height'] // 2]
    game_state['ball_direction'] = [1, 1]
    normalize_ball_direction()
    game_state['human_score'] = 0
    game_state['ai_score'] = 0
    game_state['game_active'] = True
    return jsonify(game_state)

@app.route('/update', methods=['POST'])
def update():
    if not game_state['game_active']:
        return jsonify(game_state)
    
    data = request.get_json()
    human_move = data.get('move', None)
    update_game_state(human_move)
    return jsonify(game_state)

if __name__ == '__main__':
    os.makedirs('static/models', exist_ok=True)
    if platform.system() == 'Windows':
        try:
            from waitress import serve
            print("Running on http://127.0.0.1:5000")
            serve(app, host='127.0.0.1', port=5000)
        except ImportError:
            print("Waitress not installed. Using Flask's development server instead.")
            app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        app.run(debug=True, host='127.0.0.1', port=5000)
