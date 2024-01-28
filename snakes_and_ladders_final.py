# -*- coding: utf-8 -*-
# pylint: disable=E0012, fixme, invalid-name, no-member, W0102, W0612, R0914, R0913, R1716, R1723, W0613, W0622, E0611, W0603, R0902, R0903, C0301
"""
@author: Yasmin Bettles;
Student ID: 2158064
Snakes and Ladders
Scientific Computing Y3
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

DEFAULT_SPACES, DEFAULT_SNAKES, DEFAULT_LADDERS, DEFAULT_WIN_COND, DEFAULT_DICE = 101, 10, 10, "post", 6 # 101 spaces by default bc inlcuding zero
DEFAULT_START = np.zeros(DEFAULT_SPACES)
DEFAULT_START[0] = 1
DEFAULT_SL_DICT = {1:38, 4:14, 9:31, 16:6, 21:42, 28:84, 36:44, 47:26, 49:11,
                   51:67, 56:53, 62:19, 64:60, 71:91, 80:100, 87:24, 93:73,
                   95:75, 98:78}


def gen_board_matrix(spaces=DEFAULT_SPACES, win_cond=DEFAULT_WIN_COND,
                     sl_dict=DEFAULT_SL_DICT, dice=DEFAULT_DICE):
    """
    Generates the transition matrix with the probabilities for both the dice
    behaviour and the snakes/ladders behaviour. Considers two different win conditions.
    Can change the number of side on the die, number of snakes, number of ladders.

    Parameters
    ----------
    spaces : integer, optional
        Number of spaces in the board. The default is DEFAULT_SPACES; includes space zero.
    win_cond : string, optional
        Takes values "post" or "bounce". Post win condition requires the player
        reach end space or greater; bounce requires the player to reach EXACTLY
        the end space, and if they voershoot, they are "bounced back" by however
        many spaces they overshot. The default is DEFAULT_WIN_COND.
    sl_dict : dictionary, optional
        Key:value pairs where the key is the beginning space of the snake or
        ladder, and the value is the end space of the same snake or ladder.
        The default is DEFAULT_SL_DICT.
    dice : integer, optional
        Number of sides on the dice used. The default is DEFAULT_DICE.

    Returns
    -------
    board : 2d numpy array
        Transition matrix describing probabilities of moving from one space to
        another, accounting for both dice rolls, and snakes/ladders.

    """
    board = np.zeros((spaces, spaces))

    # populate the matrix with 1/dice_sides for first "dice" number of spaces
    # after the current space (ie after leading diagonal)
    for i in range(0, spaces):
        # in row i (ie row corresponding to current space)
        # for each destination space up to i+dice (the +1 accounts for the zero-based indexing)
        # setting the corresponding probability equal to 1/dice (ie 1/6 by default)
        board[i][i+1:(i+1+dice)] = 1/dice

        if win_cond == "post":
            if i+dice >= spaces: # if less than one dice roll from end, higher prob of getting last space ("post" win condition)
                spaces_to_end = (spaces- 1) - i -1
                multiplier = dice - spaces_to_end
                prob = multiplier/dice
                if prob > 1: # if reached end space, prob's value will be >1 so set =1
                    prob = 1
                board[i][spaces-1] = prob
        elif win_cond == "bounce":
            if i+dice > spaces-1: # bounce back
                overshoot = (i+dice) - spaces
                for j in range(overshoot+1): # range is non-inclusive at top end hence +1
                    board[i][spaces-j-2] = board[i][spaces-j-2] + 1/dice

        else:
            print("if this line of code is running, something went wrong with the win condition. check your spelling?")

        # for all the snake/ladder spaces
        for key in sl_dict:
            # if a snake/ladder start space is between i+1 (dont need to check i as that's the current space) and i+dice,
            if key <= i+dice and key >= i:
                # set the end space's probability to be the previous probability + 1/dice
                # (prob of landing on start space gets added to prob of landing on end space so that the snake/ladder's effect is automatically accounted for)
                board[i][sl_dict[key]] = board[i][sl_dict[key]]+board[i][key]
                # set the start space's probability to 0
                board[i][key] = 0

    # for each snake and ladder, replace that row in the matrix with all zeroes,
    # except for the end of the ladder or snake, which should be a 1
    for key in sl_dict:
        # key = start of snake or ladder
        # value (=sl_dict[key]) = end of snake or ladder
        new_row = np.zeros(spaces)
        new_row[sl_dict[key]-1] = 1 # minus 1 inside [] because otherwise index out of bounds
        # replace the corresponding row in the matrix with one whose elements are
        # all zeroes, except for the end of the snake or ladder, which is equal
        # to 1, as the snake or ladder transports the player to the end of it 100% of the time
        board[key] = new_row
    # return the generated board matrix
    return board


def gen_random_sl_dict(snakes=DEFAULT_SNAKES, ladders=DEFAULT_LADDERS, spaces=DEFAULT_SPACES):
    """
    Allows a random set of snakes and ladders to be generated, under the
    conditions that are provided as parameters.

    Parameters
    ----------
    snakes : integer, optional
        Number of snakes. The default is DEFAULT_SNAKES.
    ladders : integer, optional
        Number of ladders. The default is DEFAULT_LADDERS.
    spaces : integer, optional
        Number of spaces on the board. The default is DEFAULT_SPACES.

    Returns
    -------
    sl : dictionary
        Key:value pairs where key is start of snake/ladder,
        and value is end of snake/ladder. If key>value it's a snake,
        if value>key it's a ladder. Key cannot equal value.

    """
    sl_dict = {} # empty dictionary for the snakes and ladders
    # generate snakes
    for i in range(snakes+1):
        space_1 = 1
        space_2 = 1
        # check that the snake doesn't already exist and doesnt take you to the same square (this doesnt quite work???)
        while np.isin(np.array([k, v] for k, v in sl_dict.items()).flatten, space_1) or np.isin(np.array([k, v] for k, v in sl_dict.items()).flatten, space_2) or space_1 == space_2:
            # if it does exist or does take you to the same square
            # (which it does at first so that it will run this line at least once)
            # then re-generate the spaces and try again until it's ok
            space_1, space_2 = np.random.randint(1, spaces-1, 2)
        # is first space higher? then it's already a snake, est key as space 1 and value as space 2
        if space_1 > space_2:
            sl_dict[space_1] = space_2
        # as equality is already accounted for in the while loop, the only other condition is
        # space 2 > space 1 so else will catch this, and in which case set the key as space 2 and value as space 1
        else:
            sl_dict[space_2] = space_1
    # repeat snake algorithm but with opposite condition on setting higher/lower space into key/value respectively
    for j in range(ladders+1):
        space_1 = 1
        space_2 = 1

        while np.isin(np.array([k, v] for k, v in sl_dict.items()).flatten, space_1) or np.isin(np.array([k, v] for k, v in sl_dict.items()).flatten, space_2) or space_1 == space_2:
            space_1, space_2 = np.random.randint(1, spaces-1, 2)
        if space_1 < space_2:
            sl_dict[space_1] = space_2
        else:
            sl_dict[space_2] = space_1
    # return the generated snake and ladder dictionary where key = start space, value = end space
    return sl_dict


def gameloop(dice=DEFAULT_DICE, win_pos=(DEFAULT_SPACES-1), sl_dict=DEFAULT_SL_DICT, win_cond=DEFAULT_WIN_COND, repeats=1, start_pos=DEFAULT_START):
    """
    Iterates through simulation of snakes and ladders game and returns the duration
    of the game (in number of turns) for each iteration of the game.

    Parameters
    ----------
    dice : integer, optional
        Number of sides on dice. The default is DEFAULT_DICE.
    win_pos : integer, optional
        Winning space to be reached by the player. The default is DEFAULT_SPACES-1
    sl_dict : dictionary, optional
        Key:value pairs where the key is the beginning space of the snake or
        ladder, and the value is the end space of the same snake or ladder.
        The default is DEFAULT_SL_DICT.
    win_cond : string, optional
        Takes values "post" or "bounce". Post win condition requires the player
        reach end space or greater; bounce requires the player to reach EXACTLY
        the end space, and if they voershoot, they are "bounced back" by however
        many spaces they overshot. The default is DEFAULT_WIN_COND.
    repeats : integer, optional
        Number of times to iterate the game loop. The default is 1.
    start_pos : array, optional
        Default starting position array. This is a row vector. The default is DEFAULT_START.

    Returns
    -------
    turns_array : 1d array
        Array of number of turns taken to win the game in each iteration.
        Its length = repeats.

    """
    # array that stores how many turns it took to win each game
    turns_array = np.zeros((repeats))
    # repeats is how many times to run the game simulation
    for i in range(repeats):
        # turns counts how many moves it takes for the player to get to the end
        turns = 0
        # reset the player's position at the start of each game
        pos = start_pos
        while np.nonzero(pos)[0] != win_pos: # while the player hasn't won
            turns += 1 # increment number of turns taken
            dice_roll = np.random.randint(1, dice+1)
            new_pos = dice_roll + np.nonzero(pos)[0]
            if new_pos < win_pos:
                # valid dice roll
                pos = np.roll(pos, dice_roll)
            elif new_pos == win_pos:
                # winning roll
                break
            else:
                # dice roll too large: handle depending on win condition
                if win_cond == "post":
                    # they win
                    break
                elif win_cond == "bounce":
                    overshoot = new_pos - win_pos
                    pos = np.roll(pos, -overshoot)
                else:
                    print("something went wrong with the win condition. check your spelling of \'post\' or \'bounce\'.")
            # check if landed on snake or ladder
            pos_index = np.nonzero(pos)[0][0]
            if pos_index in sl_dict:
                pos = np.roll(start_pos, sl_dict[pos_index]) # changing the pos to be the end of the snake/ladder if applicable
            # once the player wins, add the no. of turns it took to the turns array
        turns_array[i-1] = turns
    return turns_array


def get_fundamental(board, spaces=DEFAULT_SPACES):
    """
    Calculates fundamental matrix from the transition matrix representing the board.

    Parameters
    ----------
    board : 2d array
        Transition matrix representing the board.
    spaces : integer, optional
        Number of spaces on the board. The default is DEFAULT_SPACES.

    Returns
    -------
    fundamental : 2d array
        Fundamental matrix from the given transition matrix.

    """
    # identity matrix
    identity = np.identity(spaces-1)
    # get submatrix containing all transient states
    Q = board[0:spaces-1:, 0:spaces-1:]
    # calculate fundamental matrix (not sure if this is doing what I want it to actually)
    fundamental = np.linalg.matrix_power(identity-Q, -1)

    return fundamental


def entropy(board):
    """
    Calculates Shannon entropy for given board.

    Parameters
    ----------
    board : 2d array
        Transition matrix representing board.

    Returns
    -------
    H : float
        Shannon entropy in natural units of information.

    """
    H = 0
    #iterate through the spaces
    for row in board:
        for p in row:
            # for all non-zero probabilities,
            if p != 0:
                # sum the entropy for each space
                H += -p * np.log(p)
    return H


def heatmap(board, label):
    """
    Plots heatmaps of the number of times each spaces is landed on, on average,
    in one game of snakes and ladders.

    Parameters
    ----------
    board : 2d matrix
        Transition matrix representing the board.
    label : string
        Contains any extra information about the heatmap, to be included in
        plot title. e.g. "bounce condition" or "for 8 sided die"

    Returns
    -------
    None.

    """
    fundamental = get_fundamental(board)
    # get the first row of the fundamental matrix (in the reverse order)
    heatmap_data = fundamental[0::-1].reshape(10, 10)

    # for every other row,
    for i in range(10):
        if i%2 == 1:
            # flip the row
            heatmap_data[i] = np.flip(heatmap_data[i])

    # creating an array of numbers 1-100 to label the squares with
    space_labels = np.arange(1, 101)[::-1].reshape(10, 10)
    # for every other row,
    for i in range(10):
        if i%2 == 1:
            # flip the row
            space_labels[i] = np.flip(space_labels[i])
    # create figure
    fig = plt.figure(figsize=(8, 8))
    plt.title("Heat map for " + label)
    # plot heatmap
    heat_map = plt.imshow(heatmap_data, cmap="plasma")
    # turn off axes
    plt.axis("off")
    # changing the colour of the text over threshold value (0.6) so that it's readable
    for i in range(10):
        for j in range(10):
            if heatmap_data[i][j] > 0.6:
                colour = "black"
            else:
                colour = "lightgray"
            # labelling each square
            heat_map.axes.text(j, i, space_labels[i][j], color=colour, ha="center", va="center")
    # adding the colour bar to the heat map so that you can see what the colours represent
    plt.colorbar(heat_map)
    plt.show()


def entropy_plot(board, label):
    """
    Calculates and plots Shannon entropy for a given game board matrix.

    Parameters
    ----------
    board : 2d array
        Transition matrix of the game board for which the entropy is to be calculated and plotted.
    label : string
        Contains any details to be included in the title of the plot e.g. "bounce condition".

    Returns
    -------
    None.

    """
    ent = np.zeros(101)
    turn = np.arange(101)
    for i in turn:
        ent[i] = entropy(np.linalg.matrix_power(board, i)) # calling entropy calculating function

    max_ent = np.argmax(ent)
    fig = plt.figure(figsize=(15, 5))
    plt.title("shannon entropy against no of turns for default 100x100 board - "+label)
    plt.plot(turn, ent, color="purple", label="shannon entropy in nats")
    plt.axvline(max_ent, label="maximal entropy at "+str(np.around(max_ent))+" turns", color="red")
    plt.xlabel("turns")
    plt.ylabel("Shannon entropy")
    plt.legend()
    plt.grid()
    plt.show()


def histogram(data, title):
    """
    Plots a histogram of duration (in turns) of game, from given data. The
    title is to allow the title of the plot to be customised to reflect the data being plotted.

    Parameters
    ----------
    data : array
        The data to be presented in the histogram.
    title : string
        String to be placed in the title of the histogram.
    avg: float, optional
        The average duration as calculated using the Markov chains.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(15, 5))
    plt.title(title)
    n, bins, patches = plt.hist(data, bins="auto", density=True, color="darkviolet", label="Turns")
    mode = (bins[np.argmax(n)] + bins[np.argmax(n)+1])/2
    plt.axvline(mode, color="green", label="mode:"+str(mode))
    plt.ylabel("frequency density")
    plt.xlabel("turns")
    plt.grid()
    plt.legend()
    plt.show()


# generate random snakes and ladders
sl = DEFAULT_SL_DICT
# generating the transition matrix for the set of snakes and ladders generated
# (if sl_dict is omitted, it will default to the pre-determined board in the project pdf)
game_board_bounce = gen_board_matrix(sl_dict=sl, win_cond="bounce")
game_board_post = gen_board_matrix(sl_dict=sl) # no need to specify win condition as it's post by default


# heatmap for post win
heatmap(game_board_post, "post win condition")
# heatmap for bonuce win
heatmap(game_board_bounce, "bounce win condition")


# shannon entropy plot for post win
entropy_plot(game_board_post, "post win condition")
# shannon entropy plot for bounce win
entropy_plot(game_board_bounce, "bounce win condition")


# running the game simulation
data_post = gameloop(repeats=10000)
data_bounce = gameloop(repeats=10000, win_cond="bounce")


# plotting histogram of no. of turns taken to win snakes and ladders
histogram(data_post, "No. of turns taken to win game with post win condition")
histogram(data_bounce, "No. of turns taken to win game with bounce win condition")


# everything below this point all makes a plot of what the board looks like,
# in a separate window so it can easily be referred to with the plots

class MainWindow(QMainWindow):
    ''' the main window that holds the widget objects (eg board)'''

    def __init__(self):
        super().__init__()
        self.resize(500, 500)
        self.move(300, 300)
        central_widget = CentralWidget(self)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('Snakes & Ladders')


class CentralWidget(QWidget):
    ''' everything in the main area of the main window '''

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        # define figure canvas
        self.mpl_widget = MplWidget(self.main_window)
        # place MplWidget into a vertical box layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.mpl_widget)  # add the figure
        # use the box layout to fill the window
        self.setLayout(vbox)


class MplWidget(FigureCanvas):
    ''' both a QWidget and a matplotlib figure '''

    def __init__(self, main_window, parent=None, figsize=(4, 4), dpi=100):
        self.main_window = main_window
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.DEFAULT_SPACES = 101
        self.sl = sl
        self.plot_board(self.sl) # call plot board method
        #print(self.sl) # printing s&l dictionary to console to double check

    def plot_board(self, sl_dict, spaces=DEFAULT_SPACES-1):
        ''' plots the snakes and ladders board using matplotlib'''
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.axis("off")
        # the offset and scale to make sure that the numbers are in the right places
        offset = 0.185
        scale = 14.4
        for i in np.arange(0, 11):
            # plotting grid squares
            self.ax.axvline(i, 0.05, 0.95, color="black")
            self.ax.axhline(i, 0.05, 0.95, color="black")
            for j in np.arange(0, 11):
                if i <= 9 and j <= 9:
                    j_prime = j
                    if np.mod(i, 2) == 1: # flipping every other row
                        j_prime = 9-j # flipping the "units"
                    space = 10*i + j_prime+1
                    # putting the number text on the board
                    self.fig.text((j)/scale+offset, (i)/scale+offset, str(space))
        # plotting the snakes and ladders
        for k in sl_dict:
            # start and end of snake/ladder
            start = k-1
            end = sl_dict[k]-1
            colour = "blue" # ladders blue
            if start > end: # snakes red
                colour = "red"
            # working out coords on plot for each snake/ladder start/end space
            start_coords = [np.mod(start, 10)-0.5, np.floor(start/10)-0.5]
            end_coords = [np.mod(end, 10)-0.5, np.floor(end/10)-0.5]
            # for the rows where the numbers are "flipped", flip the position for snake/ladder
            if np.mod(np.floor(start/10), 2) == 1:
                start_coords[0] = 9-np.mod(start, 10)-0.5
            if np.mod(np.floor(end/10), 2) == 1:
                end_coords[0] = 9-np.mod(end, 10)-0.5
            # offset makes it plot in the right place
            sl_offset = 1
            y_coords = [start_coords[1]+sl_offset, end_coords[1]+sl_offset]
            x_coords = [start_coords[0]+sl_offset, end_coords[0]+sl_offset]
            # scatter dots for the start points
            self.ax.scatter(start_coords[0]+sl_offset, start_coords[1]+sl_offset, color=colour)
            # plot lines from start to end point
            self.ax.plot(x_coords, y_coords, color=colour)


app = None

# run the window that plots the board
def main():
    ''' runs the app where the snakes and ladders board is plotted'''
    global app
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == '__main__':
    main()
