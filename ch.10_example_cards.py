
import random

class DeckOfCards:
    def __init__(self):
        self.faces = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.cards = [{'face': face, 'suit': suit} for face in self.faces for suit in self.suits]

    def print_deck(self):
        for card in self.cards:
            print(f"{card['face']} of {card['suit']}", end=", ")


          
if __name__ == "__main__":
    deck_of_cards = DeckOfCards()

    print("List of 52 Playing Cards:")
    deck_of_cards.print_deck()








