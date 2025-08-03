import time
import sys
import threading
import os

start = time.time()

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


frames = [
                                                r"""
                                                     /\_/\  
                                                --> ( o.o ) <--
                                                     > - < 
                                                """,
                                                r"""
                                                     /\_/\  
                                               -->  ( -.- )   <--
                                                     > - <
                                                """,
                                                r"""
                                                     /\_/\  
                                                --> ( o.o ) <--
                                                     > - < 
                                                """,
                                                r"""
                                                     /\_/\  
                                               -->  ( ^.^ )   <--
                                                     > - <
                                                """,
                                                r"""
                                                     /\_/\  
                                                --> ( o.O ) <--
                                                     > - < 
                                                """,
                                                r"""
                                                     /\_/\  
                                                --> ( O.o ) <--
                                                     > - < 
                                                """,
                                                r"""
                                                     /\_/\  
                                               -->  ( ;.; )   <--
                                                     > - <
                                                """,
                                                r"""
                                                     /\_/\  
                                                --> ( O.O ) <--
                                                     > - < 
                                                """,
                                                r"""
                                                     /\_/\  
                                               -->  ( $.$ )   <--
                                                     > - <
                                                """,]

bye = [

        r'''
                    
                                          /^--^\     /^--^\     /^--^\
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                    BYE                 |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                    BYE                 |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                    BYE                 |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',]
from colorama import init, Fore, Style
init(autoreset=True)

def startup_animation():
    
    i = 0
    while not done:
        clear()
        print(Fore.GREEN + "Starting up please wait, here's a cat for the meanwhile :D")
        print(frames[i % len(frames)])
        time.sleep(0.3)
        i += 1

done = False
t = threading.Thread(target=startup_animation)
t.start()
import pyperclip
import math
import sympy as sp
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
from numpy import log as nplog, sin, cos, tan, exp, sqrt
import re
from fractions import Fraction
import matplotlib.pyplot as plt
import webbrowser
import urllib.parse
import tempfile
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtCore import Qt
from sympy import pretty_print

import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import traceback

import tkinter as tk

from art import text2art
from art import art

import warnings

from PyQt5.QtWidgets import QWidget, QGridLayout, QLineEdit, QPushButton, QApplication
from sympy import Matrix
import sys

done = True
t.join()

clear()
print("Ready to go!")
time.sleep(1)

def clear_screen_and_history():

        os.system('cls')

from sympy import nsimplify, Rational


# gui calc

# Remove root window creation and click function from here

# Define start_move, stop_move, do_move functions here as they depend on root, will move inside block

# Remove global variables current_input, num1, operator from here

# They will be defined inside the algebra operation block

        
class MatrixInput(QWidget):
    def __init__(self, rows, cols, title):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.setWindowTitle(title)
        self.fields = []
        self.matrix_data = None
        layout = QGridLayout()

        for i in range(rows):
            row = []
            for j in range(cols):
                field = QLineEdit()
                field.setFixedSize(35, 35)  # square fields
                field.setAlignment(Qt.AlignCenter)  # center text

                field.setStyleSheet("""
                    QLineEdit {
                        border: 2px solid black;
                        font-size: 16px;
                        background-color: #ffffff;
                    }
                """)
                layout.addWidget(field, i, j)
                layout.setHorizontalSpacing(5)
                layout.setVerticalSpacing(5)
                layout.setContentsMargins(10, 10, 10, 10)


                row.append(field)
            self.fields.append(row)

        btn = QPushButton("Submit")
        btn.clicked.connect(self.submit)
        layout.addWidget(btn, rows, 0, 1, cols)
        self.setLayout(layout)

    def submit(self):
        from sympy import Rational
        self.matrix_data = []
        for row in self.fields:
            parsed_row = []
            for cell in row:
                text = cell.text().strip()
                try:
                    parsed_row.append(Rational(text) if text else Rational(0))
                except:
                    parsed_row.append(Rational(0))
            self.matrix_data.append(parsed_row)
        self.close()


def launch_matrix_gui(rows, cols, title="Matrix"):
    app = QApplication(sys.argv)
    win = MatrixInput(rows, cols, title)
    win.show()
    app.exec_()
    return win.matrix_data



def force_stacked_str(x):
    if isinstance(x, Rational) and x.q != 1:
        n, d = str(x.p), str(x.q)
        bar = '-' * max(len(n), len(d))
        n = n.rjust(len(bar))
        d = d.rjust(len(bar))
        return f"{n}\n{bar}\n{d}"
    else:
        return str(x)

def format_cell(x, width):
    """Convert a rational number into a 3-line stacked string block."""
    if isinstance(x, Rational) and x.q != 1:
        num = str(x.p).rjust(width)
        bar = '-' * width
        den = str(x.q).rjust(width)
        return [num, bar, den]
    else:
        # Pad empty lines above and below for uniformity
        val = str(x).rjust(width)
        return [' ' * width, val, ' ' * width]

def print_pretty_matrix(matrix):
    matrix = nsimplify(matrix, rational=True)
    rows = matrix.tolist()

    # Determine max width of any numerator or denominator
    all_rationals = [x for row in rows for x in row]
    width = max(len(str(x.p)) if isinstance(x, Rational) else len(str(x)) for x in all_rationals)

    block_rows = []
    for row in rows:
        # Convert each cell to a 3-line block
        blocks = [format_cell(x, width) for x in row]
        # Transpose and join horizontally
        for i in range(3):
            block_rows.append('   '.join(cell[i] for cell in blocks))

def get_writable_locations():
    """Return possible writable locations in order of preference"""
    locations = [
        Path("desmos_graph.html"),  # 1. Current directory
        Path(tempfile.gettempdir()) / "desmos_graph.html",  # 2. System temp directory
        Path.home() / "desmos_graph.html",  # 3. User home directory
        Path.home() / "Desktop" / "desmos_graph.html"  # 4. Desktop (if exists)
    ]
    return locations


def write_desmos_html(expr_str, x_min, x_max, y_min=None, y_max=None, filename=None):
    """
    Improved version that:
    1. Shows loading status
    2. Provides clear error messages
    3. Uses a more reliable loading method
    """
    # First try to save to a temporary file
    temp_dir = Path(tempfile.gettempdir())
    filepath = temp_dir / "desmos_graph.html"
    
    # Prepare the expression
    expr_js = (expr_str.replace('\\', '\\\\')
                       .replace('`', '\\`')
                       .replace('$', '\\$'))

    # Prepare viewport bounds
    bounds_config = []
    if x_min is not None and x_max is not None:
        bounds_config.append(f"left: {x_min}, right: {x_max}")
    if y_min is not None and y_max is not None:
        bounds_config.append(f"bottom: {y_min}, top: {y_max}")
    bounds_js = f"calculator.setMathBounds({{{', '.join(bounds_config)}}});" if bounds_config else ""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Desmos Graph</title>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
        }}
        #container {{
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
        }}
        #status {{
            padding: 10px;
            background: #f0f0f0;
            text-align: center;
        }}
        #calculator {{
            flex-grow: 1;
            border: none;
        }}
        .error {{
            color: red;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="status">Loading Desmos Calculator...</div>
        <div id="calculator"></div>
    </div>

    <script>
        // Load Desmos API with callback
        function loadDesmos() {{
            const script = document.createElement('script');
            script.src = 'https://www.desmos.com/api/v1.7/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6';
            script.onload = initializeCalculator;
            script.onerror = function() {{
                document.getElementById('status').innerHTML = 
                    '<div class="error">Failed to load Desmos API. Please check your internet connection.</div>';
            }};
            document.head.appendChild(script);
        }}

        function initializeCalculator() {{
            try {{
                document.getElementById('status').innerHTML = 'Desmos loaded successfully!';
                
                const elt = document.getElementById('calculator');
                const calculator = Desmos.GraphingCalculator(elt, {{
                    keypad: true,
                    expressions: true,
                    settingsMenu: true,
                    zoomButtons: true
                }});

                // Set the expression
                calculator.setExpression({{
                    id: 'graph1',
                    latex: `{expr_js}`,
                    color: Desmos.Colors.BLUE
                }});

                // Set bounds if specified
                {bounds_js}

            }} catch (e) {{
                document.getElementById('status').innerHTML = 
                    `<div class="error">Error: ${{e.message}}<br>Expression: {expr_js}</div>`;
            }}
        }}

        // Start loading
        loadDesmos();
    </script>
</body>
</html>"""

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in default browser
        webbrowser.open(f"file://{filepath.resolve()}")
        return True
    except Exception as e:
        print(f"Error saving graph: {e}")
        return False

# Extended trig functions
def sec(x):
    return 1 / np.cos(x)

def cosec(x):
    return 1 / np.sin(x)

def cot(x):
    return 1 / np.tan(x)
    
#defines what will be the default variable
x = sp.symbols('x')

#some common functions
locals_dict = {
    'x': x,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'log': sp.log,
    'exp': sp.exp,
    'sqrt': sp.sqrt,
}

#preprocessor fixer
function_map = {
    's*in*': 'sin',
    'c*os*': 'cos',
    't*an*': 'tan',
    'c*ot*': 'cot',
    'se*c*': 'sec',
    'co*sec*': 'cosec',
    'l*og*': 'log'
}

# Preprocessor: more user-friendly
def preprocess_expr(expr_str):
    # 1. Apply function_map replacements (fix partial/truncated function names)
    for incorrect, correct in function_map.items():
        expr_str = expr_str.replace(incorrect, correct)

    # 2. Replace '^' with '**'
    expr_str = expr_str.replace('^', '**')

    # 3. Insert '*' for implicit multiplication (but avoid breaking function names!)
    # Insert '*' between digit and letter or '('
    expr_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr_str)

    # Insert '*' between ')' and letter or '('
    expr_str = re.sub(r'(\))([a-zA-Z\(])', r'\1*\2', expr_str)


    # 4. Replace sec, cosec, cot with base trig functions
    expr_str = re.sub(r'\bsec\((.*?)\)', r'1/cos(\1)', expr_str)
    expr_str = re.sub(r'\bcosec\((.*?)\)', r'1/sin(\1)', expr_str)
    expr_str = re.sub(r'\bcot\((.*?)\)', r'1/tan(\1)', expr_str)

    # 5. Handle log base syntax and ln
    expr_str = re.sub(r'log(\d+)\((.*?)\)', r'log(\2, \1)', expr_str)
    expr_str = expr_str.replace('ln(', 'log(')

    return expr_str

# Formatting for ease of reading large numbers 
def format_indian_currency(num):
    # Split the number into integer and decimal parts
    if isinstance(num, float):
        integer_part, decimal_part = str(num).split('.')
    else:
        integer_part = str(num)
        decimal_part = ''
    
    # Reverse the integer part for easier processing
    integer_part = integer_part[::-1]
    parts = []
    
    # Split the number into parts
    for i in range(0, len(integer_part), 2):
        if i == 0:
            parts.append(integer_part[i:i+3])  # First three digits
        else:
            parts.append(integer_part[i:i+2])  # Subsequent pairs of digits
    
    # Join the parts with commas and reverse back
    formatted_integer = ','.join(parts)[::-1]
    
    # Append the decimal part if it exists
    if decimal_part:
        formatted_integer += '.' + decimal_part
    
    return formatted_integer
    
#3D Graphing

def optimized_3d_discontinuity(X, Y, Z, expr_str):
    nan_mask = np.isnan(Z)
    if np.any(nan_mask):
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(nan_mask, iterations=2)

        dz_dx = np.zeros_like(Z)
        dz_dy = np.zeros_like(Z)
        dz_dx[:, 1:-1] = np.where(mask[:, 1:-1], np.abs(Z[:, 2:] - Z[:, :-2]), 0)
        dz_dy[1:-1, :] = np.where(mask[1:-1, :], np.abs(Z[2:, :] - Z[:-2, :]), 0)

        combined_grad = np.sqrt(dz_dx**2 + dz_dy**2)
        valid_gradients = combined_grad[combined_grad > 0]
        if len(valid_gradients) > 0:
            log_threshold = np.percentile(np.log(valid_gradients + 1e-10), 90)
            Z[combined_grad > np.exp(log_threshold)] = np.nan

    if 'tan(' in expr_str:
        Y_near_singular = np.abs(np.abs(Y) % np.pi - np.pi/2) < 0.08
        Z[Y_near_singular] = np.nan
    if 'log(' in expr_str:
        Z[X <= 0] = np.nan

    return Z

def plot_3d_graph(expr_str, x_range, y_range):
    x, y = sp.symbols('x y')
    expr = sp.sympify(expr_str, locals={'x': x, 'y': y})

    resolution = 80 if len(expr_str) < 50 else 40
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    f_np = sp.lambdify((x, y), expr, modules=[
        {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
         'sec': lambda v: 1 / np.cos(v),
         'cosec': lambda v: 1 / np.sin(v),
         'cot': lambda v: 1 / np.tan(v),
         'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt},
        'numpy'])

    with np.errstate(all='ignore'):
        Z = f_np(X, Y)
        Z = np.where(np.isinf(Z), np.nan, Z)
        
 
    Z = optimized_3d_discontinuity(X, Y, Z, expr_str)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Replace NaNs with mean of valid Z values for smooth rendering
    valid_z = Z[np.isfinite(Z)]
    fill_value = np.mean(valid_z) if valid_z.size > 0 else 0
    Z = np.nan_to_num(Z, nan=fill_value)

    # Normalize X and Y for centered display
    x_vals_norm = x_vals - np.mean(x_vals)
    y_vals_norm = y_vals - np.mean(y_vals)

    app = QtWidgets.QApplication([])

    view = gl.GLViewWidget()
    view.setWindowTitle(f'3D Graph of {expr_str}')
    view.setGeometry(100, 100, 800, 600)
    view.show()
    view.setCameraPosition(distance=40)

    surface = gl.GLSurfacePlotItem(
        x=x_vals_norm,
        y=y_vals_norm,
        z=Z,
        shader='shaded',
        smooth=True,
        color=(0.2, 0.6, 1, 1)
    )
    surface.setGLOptions('opaque')
    view.addItem(surface)

    QtWidgets.QApplication.instance().exec_()
    
def generate_desmos_3d_html(expr):
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Desmos 3D Graph</title>
  <script src="https://www.desmos.com/api/v1.12/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
</head>
<body>
  <div id="calculator" style="width: 1000px; height: 700px;"></div>

  <script>
    var elt = document.getElementById('calculator');
    var calculator = Desmos.Calculator3D(elt);

    calculator.setExpression({{
      id: 'graph1',
      latex: `{expr}`
    }});
  </script>
</body>
</html>
    '''
    return html_content

def Exp(a, b, n):
    n = int(n)
    if n == 0:
        return "1"
    terms = []
    for k in range(n + 1):
        coeff = math.comb(n, k)
        term_parts = []

        # Coefficient
        if coeff != 1 or (a == 0 and b == 0):
            term_parts.append(f"{coeff}")

        # a term
        if a != 0 and (n - k) > 0:
            if (n - k) == 1:
                term_parts.append(f"{a}")
            else:
                term_parts.append(f"{a}^{n - k}")

        # b term
        if b != 0 and k > 0:
            if isinstance(b, str) and b.startswith('-'):
                b_str = b[1:]
                sign = "-" if k % 2 == 1 else ""
                if sign:
                    if term_parts:
                        term_parts[0] = sign + term_parts[0]
                    else:
                        term_parts.append(sign)
                if k == 1:
                    term_parts.append(f"{b_str}")
                else:
                    term_parts.append(f"{b_str}^{k}")
            else:
                if k == 1:
                    term_parts.append(f"{b}")
                else:
                    term_parts.append(f"{b}^{k}")

        term = "*".join(term_parts)
        terms.append(term)

    return " + ".join(terms).replace("+ -", "- ")

# Allowed operands list
operands = [
    '1) Normal Calculator',
    #'2) Trigonometric',
    #'3) Exponential',
    #'4) Logarithmic',
    '2) Calculus',
    '3) 2D Graphs',
    '4) 3D Graphs',
    '5) Binomial',
    '6) Matrix Algebra' 
]
operands_str = '\n'.join(operands)

roundy = round(time.time() - start, 2)


while True:
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    #clear_screen_and_history()#clear
    print(">>> Startup:", roundy, "seconds\n")

    print(Fore.MAGENTA + text2art("CALCULATOR\n", font ="tarty"))  # Random ASCII cat
    yay = text2art("-> by Lavya :3", font ="small")
    for line in yay.splitlines():
        print(line.rjust(80))
    
    try:
        op1 = int(input(Fore.GREEN + f"Enter operation:\n{operands_str}\n>>> " + Fore.RESET))
        if op1 not in range(1, 10): 
            clear_screen_and_history()
            print(Fore.RED + "Please Enter a Valid Operation")#error
            continue
    except ValueError:
        clear_screen_and_history()
        print(Fore.RED + "Please enter a number.")#error
        continue
        
    Num1 = None
    Num2 = None
    res = None
    res2 = None
    
    clear_screen_and_history()#clear
    
    # Algebra operations
    if op1 == 1:  # Algebra selected
        root = tk.Tk()
        root.overrideredirect(True)  # Remove the title bar

        # Initialize variables
        current_input = [""]
        num1 = [None]
        operator = [None]
        root.special_functions_visible = False
        root.special_function_buttons = []

        def start_move(event):
            root.x = event.x
            root.y = event.y

        def stop_move(event):
            root.x = None
            root.y = None

        def do_move(event):
            x = root.winfo_x() - root.x + event.x
            y = root.winfo_y() - root.y + event.y
            root.geometry(f"+{x}+{y}")

        def click(btn_text):
            if btn_text in '0123456789.':
                current_input[0] += btn_text
                entry.delete(0, tk.END)
                entry.insert(tk.END, current_input[0])

            elif btn_text in '+-x/':
                if current_input[0]:
                    num1[0] = float(current_input[0])
                    operator[0] = btn_text
                    current_input[0] += operator[0]  # append operator to display
                    entry.delete(0, tk.END)
                    entry.insert(tk.END, current_input[0])

            elif btn_text == '=':
                if current_input[0] and num1[0] is not None and operator[0] is not None:
                    try:
                        parts = current_input[0].split(operator[0])
                        if len(parts) == 2:
                            num2 = float(parts[1])
                            if operator[0] == '+':
                                result = num1[0] + num2
                            elif operator[0] == '-':
                                result = num1[0] - num2
                            elif operator[0] == 'x':
                                result = num1[0] * num2
                            elif operator[0] == '/':
                                result = num1[0] / num2 if num2 != 0 else "Error"
                            entry.delete(0, tk.END)
                            entry.insert(tk.END, str(result))
                            current_input[0] = ""
                            num1[0] = None
                            operator[0] = None
                    except:
                        entry.delete(0, tk.END)
                        entry.insert(tk.END, "Error")
                        current_input[0] = ""
                        num1[0] = None
                        operator[0] = None

            elif btn_text == 'C':
                # Backspace functionality - remove last character
                if current_input[0]:
                    current_input[0] = current_input[0][:-1]
                    entry.delete(0, tk.END)
                    entry.insert(tk.END, current_input[0])
                    # If we deleted the operator, reset operator state
                    if operator[0] and (not current_input[0] or current_input[0][-1] != operator[0]):
                        operator[0] = None
                        num1[0] = None

            elif btn_text == 'AC':
                # All clear - reset everything
                entry.delete(0, tk.END)
                current_input[0] = ""
                num1[0] = None
                operator[0] = None

            elif btn_text == 'EXIT':
                root.destroy()

            elif btn_text == 'S':
                # Toggle special functions panel
                root.special_functions_visible = not root.special_functions_visible

                if root.special_functions_visible:
                    # Show special function buttons
                    special_buttons = [
                        ('sin', 'sin('),
                        ('cos', 'cos('),
                        ('tan', 'tan('),
                        ('log', 'log('),
                        ('exp', 'exp('),
                        ('sqrt', 'sqrt(')
                    ]
                    # Create buttons in new row and column
                    for i, (text, val) in enumerate(special_buttons):
                        btn = tk.Button(root, text=text, font=("Gill Sans", 14),
                                        command=lambda v=val: insert_special(v))
                        btn.grid(row=2+i, column=7, sticky='nsew', padx=2, pady=2)
                        root.special_function_buttons.append(btn)
                    # Expand grid columns for special buttons
                    for i in range(len(special_buttons)):
                        root.grid_columnconfigure(i, weight=1)
                else:
                    # Hide special function buttons
                    for btn in root.special_function_buttons:
                        btn.destroy()
                    root.special_function_buttons.clear()

        def insert_special(value):
            current_input[0] += value
            entry.delete(0, tk.END)
            entry.insert(tk.END, current_input[0])

        # Bind mouse events for moving the window
        root.bind("<Button-1>", start_move)
        root.bind("<B1-Motion>", do_move)
        root.bind("<ButtonRelease-1>", stop_move)

        # Set the title of the window (not visible due to overrideredirect)
        root.title("Calculator")

        # Configure grid weights for resizing
        for i in range(4):
            root.grid_columnconfigure(i, weight=1)
        for i in range(8):  # Increased to accommodate special function row
            root.grid_rowconfigure(i, weight=1)

        # Display label
        label = tk.Label(root, text="Calculator", font=("Gill Sans", 16))
        label.grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=5)

        # Entry widget for input/output
        entry = tk.Entry(root, font=("Gill Sans", 18), borderwidth=2, relief="sunken", justify='left')
        entry.grid(row=1, column=0, columnspan=5, sticky='nsew', padx=5, pady=5)

        # Buttons layout - common calculator buttons
        buttons = [
            'C', 'AC', 'S', 'EXIT',
            '1', '2', '3', '/',
            '4', '5', '6', 'x',
            '7', '8', '9', '-',
            '.', '0', '=', '+'
        ]

        # Create buttons in a 4x5 grid starting at row 2
        row_start = 2
        col_count = 4
        for index, btn_text in enumerate(buttons):
            row = row_start + index // col_count
            col = index % col_count
            btn = tk.Button(root, text=btn_text, command=lambda text=btn_text: click(text), font=("Gill Sans", 14))
            btn.grid(row=row, column=col, sticky='nsew', padx=2, pady=2)

        # Position window about 1/3rd from top-left corner
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_offset = screen_width // 3
        y_offset = screen_height // 3
        root.geometry(f"+{x_offset}+{y_offset}")

        root.mainloop()

    #Trig operations (in degrees)
    '''
    elif op1 == 2:
        trig_op = input("Choose Trigonometric Operation:\n"
                        "1) a * sin(b)\n"
                        "2) a * cos(b)\n"
                        "3) a * tan(b)\n"
                        "4) a * cosec(b)\n"
                        "5) a * sec(b)\n"
                        "6) a * cot(b)\n>>> ")
        
        try:
            Num1 = float(Fraction(input("\nFirst Number (a)\n>>> ")))
            Num2 = float(Fraction(input("Second Number (b)\n>>> ")))
        except ValueError:
            clear_screen_and_history()
            print(Fore.RED + "Invalid input. Please enter valid numbers.")#error
            continue
        
        if trig_op == '1':
            res = Num1 * math.sin(math.radians(Num2))
        elif trig_op == '2':
            res = Num1 * math.cos(math.radians(Num2))
        elif trig_op == '3':
            if Num2 % 180 == 90:
                clear_screen_and_history()
                print(Fore.RED + "tan undefined at", Num2)#error
            else:
                res = Num1 * math.tan(math.radians(Num2))
        elif trig_op == '4':
            if Num2 % 180 == 0:
                clear_screen_and_history()
                print(Fore.RED + "cosec undefined at", Num2)#error
            else:
                res = Num1 * cosec(math.radians(Num2))
        elif trig_op == '5':
            if Num2 % 180 == 90:
                clear_screen_and_history()
                print(Fore.RED + "sec undefined at", Num2)#error
            else:
                res = Num1 * sec(math.radians(Num2))
        elif trig_op == '6':
            if Num2 % 180 == 0:
                clear_screen_and_history()
                print(Fore.RED + "cot undefined at", Num2)#error
            else:
                res = Num1 * cot(math.radians(Num2))
                
    # Exponential
    elif op1 == 3:
        ex_op = input("Choose Exponential Operation:\n"
                      "1) a*e^b\n"
                      "2) a^b\n\n>>> ")
        
        try:
            Num1 = float(Fraction(input("\nFirst Number (a)\n>>> ")))
            Num2 = float(Fraction(input("Second Number (b)\n>>> ")))
        except ValueError:
            clear_screen_and_history()
            print(Fore.RED + "Invalid input. Please enter valid numbers.")#error
            continue
        
        if ex_op == '1':
            res = Num1 * (math.e ** Num2)
        elif ex_op == '2':
            res = Num1 ** Num2

    # Logarithm
    elif op1 == 4:
        try:
            inpt = input("Default Base(e) or Custom Base?\n1) e (Natural Log)\n2) Custom Base\n>>> ")
        except ValueError:
            clear_screen_and_history()
            print(Fore.RED + "Please choose a valid option")#error
            continue

        if inpt == '2':
            try:
                Num1 = float(Fraction(input("\nBase\n>>> ")))
                Num2 = float(Fraction(input("\nArgument ex. [log(Argument)]\n>>> ")))
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid input. Please enter valid numbers.")#error
                continue

            # Logarithm with validation for no Domain Error
            if Num1 <= 0:
                print(Fore.RED + "Invalid base for logarithm (must be > 0)")
            elif Num1 == 1:
                print(Fore.RED + "Invalid base for logarithm (cannot be 1)")
            elif Num2 <= 0:
                print(Fore.RED + "Invalid argument for logarithm (must be > 0)")
            else:
                try:
                    res = math.log(Num2, Num1)
                except ValueError as err:
                    clear_screen_and_history()
                    print(Fore.RED + "Math domain error:", err)#error

        elif inpt == '1':
            try:
                Num2 = float(Fraction(input("\nArgument\n>>> ")))
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid input. Please enter a valid number.")#error
                continue

            if Num2 <= 0:
                clear_screen_and_history()
                print(Fore.RED + "Invalid argument for logarithm (must be > 0)")#error
            else:
                try:
                    res = math.log(Num2)  # Natural log (base e)
                except ValueError as err:
                    clear_screen_and_history()
                    print(Fore.RED + "Math domain error:", err)#error
        else:
            clear_screen_and_history()
            print(Fore.RED + "Invalid selection. Please choose 1 or 2.")#error  '''
                
    # Calculus
    if op1 == 2:
        while True:
            try:
                expr_str = input("Enter the expression (use 'x' as the variable)\n>>> ")
                expr_str = preprocess_expr(expr_str)
                sym_expr = sp.sympify(expr_str, locals=locals_dict)
                break
            except (sp.SympifyError, TypeError):
                clear_screen_and_history()
                print(Fore.RED + "Invalid expression. Please enter a valid mathematical expression.")#error

        operation = input("Choose operation:\n1) Indefinite Integrate\n2) Definite Integrate\n3) Differentiate\n>>> ")

        variable = sp.symbols('x')

        if operation == '1':
            # Compute the indefinite integral
            result = sp.integrate(sym_expr, variable)
            # Check if the result is a simple expression or not
            if result.has(sp.Integral):
                clear_screen_and_history()
                print(Fore.RED + f"The indefinite integral of {expr_str} cannot be expressed in a simple form.")#error
            else:
                # Convert the result to a more readable format
                pretty_result = sp.pretty(result, use_unicode=False)
                print(f"Indefinite Integral of {expr_str}:\n {pretty_result}   + C [separate term]")

        elif operation == '2':
            # Numeric definite integral using scipy
            def f_np(x):
                return sp.lambdify(variable, sym_expr, 'numpy')(x)

            try:
                lower_bound = float(input("Enter lower limit (a)\n>>> "))
                upper_bound = float(input("Enter upper limit (b)\n>>> "))
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid bounds.")#error
                continue

            numeric_integral, error = quad(f_np, lower_bound, upper_bound)
            print(f"Numerical Definite Integral from {lower_bound} to {upper_bound}: {numeric_integral} (approx.)")

        elif operation == '3':
            try:
                # Compute the derivative
                derivative = sp.diff(sym_expr, variable)
                
                # Simplify the result
                simplified = sp.simplify(derivative)
                
                # Check if the derivative could be computed
                if derivative.has(sp.Derivative):
                    clear_screen_and_history()
                    print(Fore.RED + f"The derivative of {expr_str} could not be computed in a simple form.")#error
                else:
                    # Convert to pretty format
                    pretty_result = sp.pretty(simplified, use_unicode=False)
                    
                    # Print with beautiful formatting
                    print(f"Derivative of {expr_str}:\n{pretty_result}")
                    
                    # Copy to clipboard
                    pyperclip.copy(str(simplified))
                    print("\n(Result copied to clipboard)")
                        
            except Exception as e:
                clear_screen_and_history()
                print(Fore.RED + f"Error computing derivative: {str(e)}")#error
            
    # Graphs 2D
    if op1 == 3:
        expr_str = input("Enter the function of x\n>>> f(x) = ")
        expr_str = preprocess_expr(expr_str)

        try:
            expr = sp.sympify(expr_str, locals=locals_dict)
        except (sp.SympifyError, TypeError) as e:
            clear_screen_and_history()
            print(Fore.RED + "Invalid expression:", e)#error
            continue

        # Ask which graphing method to use
        graph_method = input("Choose graphing method:\n1) Matplotlib Graphing Method (Offline) [doesnt support all functions] \n2) Desmos Graphing Method (Online) [recommended]\n>>> ")

        # Get bounds (common for both methods)
        custom_range = input("Use custom x-axis range? (y/n)\n >>> ").lower()
        if custom_range == 'y':
            try:
                lower = float(input("Enter lower bound for x\n>>> "))
                upper = float(input("Enter upper bound for x\n>>> "))
                if upper <= lower:
                    clear_screen_and_history()
                    print(Fore.RED + "Upper bound must be greater than lower bound."+ Fore.GREEN +" Using -5 to 5.")#error_1
                    lower, upper = -5, 5
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid input."+ Fore.GREEN +" Using default range (-5 to 5).")#error
                lower, upper = -5, 5
        else:
            lower, upper = -5, 5

        # Get y-bounds if desired
        custom_y = input("Use custom y-axis range? (y/n)\n>>> ").lower()
        if custom_y == 'y':
            try:
                y_lower = float(input("Enter lower bound for y\n>>> "))
                y_upper = float(input("Enter upper bound for y\n>>> "))
                if y_upper <= y_lower:
                    clear_screen_and_history()
                    print(Fore.RED + "Upper bound must be greater than lower bound." + Fore.GREEN + " Using auto-scaling.")#error_1
                    y_lower, y_upper = None, None
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid input." + Fore.GREEN + " Using auto-scaling.")#error
                y_lower, y_upper = None, None
        else:
            y_lower, y_upper = None, None
            
        if graph_method == '1':
            x_vals = np.linspace(lower, upper, 2000)  # High resolution sampling

            try:
                f_np = sp.lambdify(x, expr, modules=[{
                    'log': nplog,
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'exp': np.exp,
                    'sqrt': np.sqrt,
                    'sec': sec,
                    'cosec': cosec,
                    'cot': cot
                }, 'numpy'])

                y_vals = f_np(x_vals)

                # Handle complex results
                if np.iscomplexobj(y_vals):
                    y_vals = np.real(y_vals)
                    print(Fore.GREEN + "Note: Showing real part of complex-valued function")

                diffs = np.abs(np.diff(y_vals))
                valid_diffs = diffs[np.isfinite(diffs)]

                if len(valid_diffs) > 0:
                    threshold = np.percentile(valid_diffs, 99.85) * 10
                    jump_indices = np.where(diffs > threshold)[0]

                    pad = 3  # Points to mask around jumps
                    for i in jump_indices:
                        start = max(0, i - pad)
                        end = min(len(y_vals), i + pad + 1)
                        y_vals[start:end] = np.nan
                else:
                    jump_indices = []

                # Calculate y limits dynamically
                valid_y = y_vals[np.isfinite(y_vals)]
                if y_lower is None or y_upper is None:
                    if len(valid_y) > 0:
                        y_min = np.min(valid_y) - 1  # Add some padding
                        y_max = np.max(valid_y) + 1  # Add some padding
                    else:
                        y_min, y_max = -5, 5  # Default limits if no valid y values
                else:
                    y_min, y_max = y_lower, y_upper

                plt.figure(figsize=(10, 6))
                ax = plt.gca()

                # Plot main function line
                main_line = ax.plot(
                    x_vals, y_vals,
                    linewidth=2.5,
                    color='#1f77b4',
                    solid_capstyle='round',
                    label=f'${sp.latex(expr)}$'
                )[0]

                # Show asymptote/infinity markers
                for i in jump_indices:
                    if i < len(y_vals):
                        if y_vals[i] > 0:
                            ax.plot(x_vals[i], y_max * 1.05, 'v', color='#1f77b4', markersize=8)
                        else:
                            ax.plot(x_vals[i], y_min * 1.05, '^', color='#1f77b4', markersize=8)

                # If y-values are very large, use symlog scale gracefully
                if np.max(np.abs(y_vals[np.isfinite(y_vals)])) > 1e6:
                    plt.yscale('symlog')

                # Add asymptote approach indicators
                for i in jump_indices:
                    if i > 10 and i < len(y_vals) - 10:  # Avoid edge points
                        x_approach = x_vals[i - 5:i]
                        y_approach = y_vals[i - 5:i]

                        if np.any(np.abs(y_approach) > 100):
                            ax.plot(
                                x_approach, y_approach,
                                linewidth=3,
                                color='#ff7f0e',
                                alpha=0.7,
                                zorder=5
                            )
                            ax.annotate(
                                '',
                                xy=(x_vals[i], np.sign(y_approach[-1]) * y_max * 0.9),
                                xytext=(x_vals[i - 2], y_approach[-2]),
                                arrowprops=dict(arrowstyle="->", color='#ff7f0e', lw=1.5),
                                zorder=10
                            )

                ax.set_title(f'Graph of ${sp.latex(expr)}$', fontsize=14, pad=10)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)

                ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.axvline(0, color='black', linewidth=0.8)

                ax.set_xlim(lower, upper)
                ax.set_ylim(y_min, y_max)  # Use dynamically calculated limits

                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                ax.yaxis.set_major_locator(plt.MaxNLocator(10))

                if max(abs(y_min), abs(y_max)) > 1e3:
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax.yaxis.get_offset_text().set_fontsize(10)

                plt.legend(fontsize=10, loc='best', framealpha=0.5)
                plt.tight_layout(pad=1.5)
                plt.show()
                

            except Exception as e:
                clear_screen_and_history()
                print(Fore.RED + "Could not plot the function:", e)#error
                print(Fore.RED + "Falling back to Desmos...")#error
                time.sleep(1)
                write_desmos_html(sp.latex(expr), lower, upper, y_lower, y_upper)

        elif graph_method == '2':
            try:
                # Convert SymPy expression to LaTeX for Desmos
                latex_expr = sp.latex(expr)
                #print(f"LaTeX expression: {latex_expr}")  # Debug print
                
                # Generate Desmos HTML and open it
                print(Fore.GREEN + "Opening Desmos in your web browser...")
                write_desmos_html(
                    latex_expr,
                    x_min=lower,
                    x_max=upper,
                    y_min=y_lower,
                    y_max=y_upper
                )
                print(Fore.GREEN + "Note: Close the browser tab when done.")
            except Exception as e:
                print(Fore.RED + "\nERROR in Desmos graphing:")
                print(Fore.RED + f"Type: {type(e).__name__}")
                print(Fore.RED + f"Message: {str(e)}")
                print(Fore.RED + "\nTroubleshooting tips:")
                print("1. Check your internet connection (Desmos requires online access)")
                print("2. Try a simpler expression to test")
                print("3. Make sure your bounds are valid numbers")
                print("4. The expression may contain unsupported functions")
                
                # Fall back to matplotlib if Desmos fails
                fallback = input("Would you like to try Matplotlib instead? (y/n): ").lower()
                if fallback == 'y':
                    graph_method = '1'
                    continue  # This will restart the graphing section

# 3D graphing
    elif op1 == 4:
        expr_str = input("Enter the function of x and y (use 'x' and 'y' as variables)\n>>> f(z) = ")
        #expr_str = preprocess_expr(expr_str)

        # Ensure "z = " prefix for Desmos 3D if user selects that option
        if not expr_str.strip().lower().startswith('z=') and not expr_str.strip().lower().startswith('z ='):
            desmos_expr_str = "z = " + expr_str
        else:
            desmos_expr_str = expr_str

        # Get ranges as before
        custom_range = input("Use custom x-axis range? (y/n)\n>>> ").lower()
        if custom_range == 'y':
            try:
                lower = float(input("Enter lower bound for x\n>>> "))
                upper = float(input("Enter upper bound for x\n>>> "))
                if upper <= lower:
                    clear_screen_and_history()
                    print(Fore.RED + "Upper bound must be greater than lower bound." + Fore.GREEN + " Using default range (-5 to 5).")#error
                    lower, upper = -5, 5
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid input." + Fore.GREEN +  " Using default range (-5 to 5).")#error
                lower, upper = -5, 5
        else:
            lower, upper = -5, 5

        custom_y = input("Use custom y-axis range? (y/n)\n>>> ").lower()
        if custom_y == 'y':
            try:
                y_lower = float(input("Enter lower bound for y\n>>> "))
                y_upper = float(input("Enter upper bound for y\n>>> "))
                if y_upper <= y_lower:
                    clear_screen_and_history()
                    print(Fore.RED + "Upper bound must be greater than lower bound." + Fore.GREEN +  " Using default range (-5 to 5).")#error
                    y_lower, y_upper = -5, 5
            except ValueError:
                clear_screen_and_history()
                print(Fore.RED + "Invalid input."+ Fore.GREEN +" Using default range (-5 to 5).")#error
                y_lower, y_upper = -5, 5
        else:
            y_lower, y_upper = -5, 5

        # Choose graphing engine for 3D
        engine = input("Choose 3D graphing engine:\n1) PyQtGraph (offline) [doesnt support all functions]\n2) Desmos (online) [recommended]\n>>> ")

        if engine == '1':
            try:
                plot_3d_graph(expr_str, (lower, upper), (y_lower, y_upper))
            except Exception as e:
                clear_screen_and_history()
                print(Fore.RED + f"Error plotting 3D graph with PyQtGraph: {e}")#error
        elif engine == '2':
            try:
                html_content = generate_desmos_3d_html(desmos_expr_str)
                temp_dir = Path(tempfile.gettempdir())
                filepath = temp_dir / "desmos_3d_graph.html"
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(html_content)
                webbrowser.open("file://" + str(filepath.resolve()))
            except Exception as e:
                clear_screen_and_history()
                print(Fore.RED + "Error in 3D Desmos graphing:")#error
                print(e)
        else:
            clear_screen_and_history()
            print(Fore.RED + "Invalid selection." + Fore.WHITE + "Please choose 1 or 2.")#error
    elif op1 == 5:
        print("Enter the terms for the binomial expansion form: (a+b)^n")
        
        # Get user input for a, b, and n
        a = input("a = ")
        b = input("b = ")
        
        # Validate and convert n to a float
        while True:
            try:
                n = float(input("n = "))
                if n < 0:
                    print("Please enter a non-negative value for n.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a valid number for n.")
        
        # Calculate the binomial expansion
        res2 = Exp(a, b, n)
        
        
    elif op1 == 6:
        try:
            print("Matrix Operations:")
            print("1) Inverse")
            print("2) Addition")
            print("3) Multiplication")
            opm = int(input("Select matrix operation (1-3):\n>>> "))

            if opm not in {1, 2, 3}:
                print(Fore.RED + "Invalid matrix operation")
                continue

            rows1 = int(input("Enter number of rows for Matrix A: "))
            cols1 = int(input("Enter number of columns for Matrix A: "))

            if opm in {2, 3}:
                rows2 = int(input("Enter number of rows for Matrix B: "))
                cols2 = int(input("Enter number of columns for Matrix B: "))
            
            # Launch GUI(s)
            matA = launch_matrix_gui(rows1, cols1, "Matrix A")
            
            matB = None
            if opm in {2, 3}:
                matB = launch_matrix_gui(rows2, cols2, "Matrix B")
            
            # SymPy operations
            from sympy import Matrix
            print("\nResult:")
            
            mt = None
            if opm == 1:
                if rows1 != cols1:
                    print(Fore.RED + "Matrix must be square for inverse.")
                    continue
                mt = Matrix(matA).inv()
                A = nsimplify(mt, rational=True)
                for row in A.tolist():
                    cell_strs = [force_stacked_str(x) for x in row]
                    print("   ".join(cell_strs))

            
            elif opm == 2:
                if rows1 != rows2 or cols1 != cols2:
                    print(Fore.RED + "Matrix dimensions must match for addition.")
                    continue
                mt = Matrix(matA) + Matrix(matB)
                A = nsimplify(mt, rational=True)
                for row in A.tolist():
                    cell_strs = [force_stacked_str(x) for x in row]
                    print("   ".join(cell_strs))
            
            elif opm == 3:
                if cols1 != rows2:
                    print(Fore.RED + "Invalid dimensions for multiplication (A.columns ≠ B.rows).")
                    continue
                mt = Matrix(matA) * Matrix(matB)
                A = nsimplify(mt, rational=True)
                for row in A.tolist():
                    cell_strs = [force_stacked_str(x) for x in row]
                    print("   ".join(cell_strs))
        
        except Exception as e:
            print(Fore.RED + f"Error: {e}")
            continue
    
    # Output result
    if res is not None:
        try:
            res_rounded = round(res, 3)
            formatted_result = format_indian_currency(res_rounded)  # Format the result
            print("Result: ", formatted_result)  # Print the formatted result
            pyperclip.copy(str(res))  # Copy the unformatted numeric result to clipboard
            print("Result copied to clipboard!")
        except Exception as e:
            clear_screen_and_history()
            print(Fore.RED + "Error formatting or copying result:", e)#error

    if res2 is not None:
        try:
            resy = sp.simplify(res2)
            res_2 = sp.pretty(resy, use_unicode=False)

            print("Result: \n", res_2)  # Print the formatted result
            pyperclip.copy(str(res2))  # Copy the unformatted numeric result to clipboard
            print("Result copied to clipboard!")
        except Exception as e:
            clear_screen_and_history()
            print(Fore.RED + "Error formatting or copying result:", e)#error

    #continue the calculations
    con = input("Do you want to make another calculation? (y/n):\n").lower() 
    clear_screen_and_history()#clear
    
    done = False
    if con != 'y':
        try:
            bye = [

                r'''
                    
                                          /^--^\     /^--^\     /^--^\ zz
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                                        |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\ zzz
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                                        |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\ zzzz
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                                        |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\ zzz
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                                        |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\ zz
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                                        |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',
                    r'''
                    
                                          /^--^\     /^--^\     /^--^\ z
                                          \____/     \____/     \____/
                                         /      \   /      \   /      \
                                        |        | |        | |        |
                                         \__  __/   \__  __/   \__  __/
                    |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
                    | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
                    ########################/ /######\ \###########/ /#######################
                    | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
                    |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
                    ''',]
                                    
            j = 0
            while not done:
                clear()
                print(bye[j % len(bye)])
                time.sleep(0.4)
                j += 1
                
                if j > 6: 
                    done = True
            
            #print(Fore.CYAN + bye)
            sys.exit()
        except Exception as e:
            print(e)

