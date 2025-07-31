import math

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def is_moves_left(board):
    for row in board:
        if " " in row:
            return True
    return False

def evaluate(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] !=" ":
             return 10 if row[0] == 'O' else -10
            
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != " ":
                return 10 if board[0][col] == 'O' else -10
            
        

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != " ":
            return 10 if board[0][0] == 'O' else -10
    
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != " ":
            return 10 if board[0][2] == 'O' else -10

        
    
    return 0

def minimax(board , depth, is_max, alpha, beta):
    score =  evaluate(board)

    if score == 10:
        return score - depth
    if score == -10:
        return score + depth 
    if not is_moves_left(board):
        return 0
    
    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = 'O'
                    best = max(best, minimax(board , depth +1, False, alpha, beta))
                    board[i][j] = " "
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = 'X'
                    best = min(best, minimax(board, depth +1, True, alpha, beta))
                    board[i][j] = " "
                    best = min(beta, best)
                    if beta <= alpha:
                        break
        return best

def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board [i][j]= 'O'
                move_val = minimax(board, 0, False, -math.inf, math.inf)
                board[i][j] = " "
                if move_val > best_val:
                    best_val = move_val
                    best_move = (i, j)

    return best_move

def play_game():
    board = [[" " for _ in range(3)]for _ in range(3)]
    print("welcome to Tic-toc-roe!\n you are 'X' and the AI is 'O'.")
    print_board(board)

    while True:
        if not is_moves_left(board):
            print("It's aa draw!")
            break

        print("\n current board:")
        print_board(board)

        while True:
            try:
                row = int(input("Enter your move row(0-2): "))
                col = int(input("Enter your move column(0-2): "))
            except ValueError:
                print("enter numbers only! ")
                continue

        

            if row not in range(3) or col not in range(3):

                print("Invalid move! Enter row and column between 0 and 2.")
                continue

            if board[row][col] != " ":
                print(f"cell ({row},{col}) is already occupied by '{board[row][col]}'. try another move.")
                continue
            break

        board[row][col] = 'X'
        print("\n your Move:")
        print_board(board)

        if evaluate(board) == -10:
            print("You win !")
            break
        if not is_moves_left(board):
            print("It's a draw!")
            break

        ai_move = find_best_move(board)
        board[ai_move[0]][ai_move[1]] = 'O'
        print("\nAI's Move:")
        print_board(board)

        if evaluate(board) == 10:
            print("AI wins!")
            break 
        if not is_moves_left(board):
            print("It's a draw!")
            break 

play_game()

