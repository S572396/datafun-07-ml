import random

class DeckOfCards:
    def __init__(self):
        self.faces = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.cards = [{'face': face, 'suit': suit} for face in self.faces for suit in self.suits]

    def shuffle_deck(self):
        random.shuffle(self.cards)

    def get_shuffled_deck_string(self):
        shuffled_deck_string = ""
        for card in self.cards:
            shuffled_deck_string += f"{card['face']} of {card['suit']}, "
        return shuffled_deck_string.strip(', ')


if __name__ == "__main__":
    deck_of_cards = DeckOfCards()

    print("Original Deck:")
    print(deck_of_cards.get_shuffled_deck_string())  

    # Shuffle the deck
    deck_of_cards.shuffle_deck()

    print("\nShuffled Deck:")
    print(deck_of_cards.get_shuffled_deck_string())  





