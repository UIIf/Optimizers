from sympy import Symbol, sin, cos, lambdify, Piecewise
def task1():
    symbol_x = Symbol('x')
    func =  Piecewise(
        (-symbol_x+1, symbol_x < 0),
        (symbol_x+0, symbol_x > 1),
        (symbol_x*sin(symbol_x*10) + symbol_x*symbol_x*cos(symbol_x*30) + 1, True),
    )
    dfunc = func.diff(symbol_x)
    ddfunc = dfunc.diff(symbol_x)
    func = lambdify(symbol_x, func, 'numpy')
    dfunc = lambdify(symbol_x, dfunc, 'numpy')
    ddfunc = lambdify(symbol_x, ddfunc, 'numpy')
    
    return func, dfunc, ddfunc