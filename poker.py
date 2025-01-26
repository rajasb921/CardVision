'''
- Use this program to see how well the model can detect cards in a poker game
- The program will simulate a poker game with two players
- The first player will have a random hand detected by the model, while the second player will simply have a random hand
- The program will display the odds of each player winning and tying
- The program will also display the average error in the percentages of the guessed cards    

Texas Hold'em simulation code is from the poker_calc package
'''

import cv2
import random
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
from poker_calc.pokergame.pokergame import TexasHoldem # version: 0.0.2


# Define a function to get the odds of a player winning & tying
def get_odds(player):
    odds = round(player.wins/player.runs * 100, 2)
    tie = round(player.ties/player.runs * 100, 2)

    return odds, tie

# Define a function to get the prediction of the model
def get_prediction(hands):
    images = []
    labels = []

    # Generate labelMap
    denominations = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['C', 'H', 'S', 'D']
    labelMap = {f"{denom}{suit}": idx for idx, (denom, suit) in enumerate((d, s) for d in denominations for s in suits)}

    suits = ['c', 'h', 's', 'd']
    reverseMap = {idx: f"{denom}{suit}" for idx, (denom, suit) in enumerate((d, s) for d in denominations for s in suits)}

    img_dir = './cards/'
    for img_name in hands:
        label_name = img_name[:2]
        label = labelMap[label_name]

        # Read image
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
        img = img / 255.0

        # Slightly augment the image
        factor = random.choice([0.9, 1.1])
        img = np.clip(img * factor, 0, 1)
        h_shift = random.choice([2, -2])
        v_shift = random.choice([2, -2])
        M = np.float32([[1, 0, h_shift], [0, 1, v_shift]])
        img = cv2.warpAffine(img, M, (128, 128), flags=cv2.INTER_LINEAR)

        # Append image
        images.append(img)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    model = load_model('./mobilenet_card_classifier_model.keras')
    predictions = model.predict(images)
    predictions = np.argmax(predictions, axis=1)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    return [reverseMap[pred] for pred in predictions]

# Define a function to get the true hand
def get_true(hands):
    labels = []
    for img_name in hands:
        denomination = img_name[0]
        suit = img_name[1]
        labels.append(f"{denomination}{suit.lower()}")
    return labels

# Define a function to show the cards
def show_cards(hands, player_hand):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Your hand")
    fig.suptitle("Detected hand: " + player_hand[0] + " " + player_hand[1])
    for i, img_name in enumerate(hands):
        img_path = os.path.join('./cards/', img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()

# Define a function to generate random hands
def generate_random_hands():
    # Select 4 random cards from card images
    card_images = os.listdir('./cards/')
    random.shuffle(card_images)
    hands = card_images[:4]
    
    player1_hand = get_prediction(hands[:2])
    show_cards(hands[:2], player1_hand)
    player1_true_hand = get_true(hands[:2])
    player2_hand = get_true(hands[2:])

    return player1_hand+player2_hand, player1_true_hand

def main():
    # Track correct predictions and wrong predictions
    total = 0
    correct = 0
    avg_error = 0 # Average error in the percentages of the guessed cards

    play = True

    while play:
        # Create a TexasHoldem objects for the true game, and game with detected cards
        true_game = TexasHoldem()
        game = TexasHoldem()

        # Add players to the game, for now keep a specific hand
        hands, true_hands = generate_random_hands() # Hands = 4 cards, true_hands = 2 cards only for player 1 (Since these are detected by the model)

        game.add_player("Player 1", hands[0], hands[1])
        game.add_player("Player 2", hands[2], hands[3])
            
        true_game.add_player("Player 1", true_hands[0], true_hands[1])
        true_game.add_player("Player 2", hands[2], hands[3])

        game.run(iterations=5000)
        true_game.run(iterations=5000)

        players = game.players_list
        true_players = true_game.players_list

        p1_scores = get_odds(players[0])
        p2_scores = get_odds(players[1])

        true_p1_scores = get_odds(true_players[0])

        os.system('cls' if os.name == 'nt' else 'clear')
        print("--------------------------------------------------------------------------------------------")
        # Raise if player 1 odds are higher than 70%
        if p1_scores[0] > 70:
            print("Based on the system's prediction, you have a high chance of winning. Raise.")
        elif p1_scores[0] < 40:
            print("Based on the system's prediction, you have a low chance of winning. Fold the hand.")
        else:
            print("Based on the system's prediction, you have a moderate chance of winning. Call the bet.")
        print("---------------------------------------------------------------------------------------------")

        total += 1
        if abs(p1_scores[0] - true_p1_scores[0]) < 5: # Randomness of simulation can cause the odds to be slightly off
            correct += 1
        else:
            avg_error += abs(p1_scores[0] - true_p1_scores[0])

        print()
        print()
        print("Play again?")
        while True:
            try:
                play_again = input("1. Yes\n2. No\n")
                if play_again == "1" :
                    break
                elif play_again == "2":
                    play = False
                    break
                else:
                    print("Invalid input. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Total games: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Predictions accuracy: {round(correct/total * 100, 2)}%")
    print(f"Average error: {avg_error}%")

if __name__ == "__main__":
    main()
