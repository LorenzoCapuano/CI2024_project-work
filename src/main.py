import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from enum import Enum

#Function definition: (add, sub, mul, div, pow, sin , cos, abs, inv)

def _add(args):
    return args[0] + args[1]

def _add_print(args, _print):
    print('(', end='')
    _print(args[0])
    print('+', end='')
    _print(args[1])
    print(')', end='')

def _sub(args):
    return args[0] - args[1]

def _sub_print(args, _print):
    print('(', end='')
    _print(args[0])
    print('-', end='')
    _print(args[1])
    print(')', end='')

def _mul(args):
    return args[0] * args[1]

def _mul_print(args, _print):
    print('(', end='')
    _print(args[0])
    print('*', end='')
    _print(args[1])
    print(')', end='')

def _div(args):
    return args[0] / args[1]

def _div_print(args, _print):
    print('(', end='')
    _print(args[0])
    print('/', end='')
    _print(args[1])
    print(')', end='')

def _pow(args):
    return args[0] ** args[1]

def _pow_print(args, _print):
    print('(', end='')
    _print(args[0])
    print('**', end='')
    _print(args[1])
    print(')', end='')

def _sin(args):
    return np.sin(args[0])

def _sin_print(args, _print):
    print('np.sin(', end='')
    _print(args[0])
    print(')', end='')

def _cos(args):
    return np.cos(args[0])

def _cos_print(args, _print):
    print('np.cos(', end='')
    _print(args[0])
    print(')', end='')

def _abs(args):
    return np.abs(args[0])

def _abs_print(args, _print):
    print('np.abs(', end='')
    _print(args[0])
    print(')', end='')

def _inv(args):
    return 1 / args[0]

def _inv_print(args, _print):
    print('(1/', end='')
    _print(args[0])
    print(')', end='')

def _const(var):
    return var
def _const_print(args, _print):
    if args.name is not None:
        print(args.name, end='')
        return

    if(args.val < 0):
        print('(',args.val,')', end='', sep='')
    else:
        print(args.val, end='')

def _var(index, table):
    return table[index]

def _var_print(args, _print):
    print('x[',args,']', end='', sep='')

class OP:
    def __init__(self, op, opType, nArgs, print):
        self.nArgs = nArgs
        self.op = op
        self.opType = opType
        self.print = print

class CONST:
    def __init__(self, val, name):
        self.val = val
        self.name = name

class ops_types(Enum):
    add = 1
    sub = 2
    mul = 3
    div = 4
    pow = 5
    sin = 6
    cos = 7
    const = 8
    var = 9
    inv = 10
    abs = 11

add = OP(_add, ops_types.add, 2, _add_print)
sub = OP(_sub, ops_types.sub, 2, _sub_print)
mul = OP(_mul, ops_types.mul, 2, _mul_print)
div = OP(_div, ops_types.div, 2, _div_print)
pow = OP(_pow, ops_types.pow, 2, _pow_print)
sin = OP(_sin, ops_types.sin, 1, _sin_print)
cos = OP(_cos, ops_types.cos, 1, _cos_print)
inv = OP(_inv, ops_types.inv, 1, _inv_print)
abs = OP(_abs, ops_types.abs, 1, _abs_print)
const = OP(_const, ops_types.const, 1, _const_print)
var = OP(_var, ops_types.var,1, _var_print)

#basic building blocks available for creating an expression. I used different pools for different problems.

opsPool = [add, sub, mul, inv, cos] #[add, sub, mul, inv, sin, cos, pow, abs]
termPool = [const, var]
constPool = [CONST(0, None),
             CONST(1, None),
             CONST(-1, None),
             CONST(2, None),
             CONST(-2, None),
             CONST(np.pi, 'np.pi'),
             CONST(np.e, 'np.e')
             ]

opsPoolBynArgs = [[sin, cos, inv, abs, cos], #1
                  [add, sub, mul, div, pow] #2
                ]
    
'''
CONST(np.pi, 'pi'),
CONST(np.e, 'e')
'''


def get_random_element(l):
    return l[np.random.randint(len(l))]

class Node:
    def __init__(self, op, args, parent = None):
        self.op = op
        self.args = args
        self.parent = parent
        self.dim = 1
        self.tempRes = None

    def copy(self):
        cp = Node(self.op, self.args, self.parent)
        cp.dim = self.dim
        cp.tempRes = self.tempRes
        return cp

'''
class valFlag:
    def __init__(self, startVal, flag):
         self.startVal = startVal
         self.flag = flag
'''

class Expr:
    def __init__(self, nVar):
        self.start = None
        self.nVar = nVar
        self.constList = []
        self.fitness = None

    #initialize expression with random full
    def init_random_full(self, maxD):
        self.start = Node(get_random_element(opsPool), None, None)
        
        qa = [self.start]
        d = 0
        while(d < maxD):
            qb = []
            for n in qa:
                ch = []
                for _ in range(n.op.nArgs):    
                    if(d < maxD-1):
                        ch += [Node(get_random_element(opsPool), None, n)]
                    else:
                        termType = np.random.randint(2)
                        if(termType == 0): #const
                            ch += [Node(const, get_random_element(constPool), n)]
                        else: #var
                            ch += [Node(var, np.random.randint(self.nVar), n)]
                n.args = ch
                qb += ch
            qa = qb
            d+=1
        
        self.updateNodeDim()

    #initialize expression with random grow
    def init_random_grow(self, maxD):
        self.start = Node(get_random_element(opsPool), None, None)
        
        qa = [self.start]
        d = 0
        while(d < maxD and len(qa) > 0):
            qb = []
            for n in qa:
                ch = []
                for _ in range(n.op.nArgs):
                    nodeType = np.random.randint(2)
                    new = None
                    if(nodeType and d < maxD-1):
                        new = Node(get_random_element(opsPool), None, n)
                        ch += [new]
                        qb += [new]
                    else:
                        termType = np.random.randint(2)
                        if(termType == 0): #const
                            ch += [Node(const, get_random_element(constPool), n)]
                            
                        else: #var
                            ch += [Node(var, np.random.randint(self.nVar), n)]
                n.args = ch
            qa = qb
            d+=1

        self.updateNodeDim()

    #Evaluate expression
    def getResult(self, vars):

        def _getRes(node):

            #Reuse evaluateted subexpression
            if(node.tempRes is not None):
                return node.tempRes
            
            #print(node.op)

            res = []

            if(node.op.opType == ops_types.const):
                node.tempRes = np.ones(vars.shape[1]) * node.args.val
                return node.tempRes
        
            if(node.op.opType == ops_types.var):
                node.tempRes = vars[node.args, :]
                return node.tempRes

            for n in node.args:
                res += [_getRes(n)]

            node.tempRes = node.op.op(res)
            return node.tempRes

        return _getRes(self.start)
    
    def updateConstList(self):
        def _upcl(node):
            #print(node.op.op)
            if(node.op.opType == ops_types.const):
                #print('here')
                self.constList += [node]
                return
            
            if(node.op.opType == ops_types.var):
                return
            
            for n in node.args:
               _upcl(n)
        
        self.constList = []
        _upcl(self.start)
    

    def updateNodeDim(self):

        def _upnd(node):

            if(node.tempRes is not None):
                return node.dim

            node.dim = 1
            if(node.op.opType == ops_types.const or node.op.opType == ops_types.var):
                return 1
            
            for n in node.args:
                node.dim += _upnd(n)

            return node.dim

        _upnd(self.start)

    
    def exchange_random_subtrees(self, parent):
        if(self.start.dim == 1 or parent.start.dim == 1):
            return (self, parent)

        i1 = np.random.randint(1, self.start.dim)
        i2 = np.random.randint(1, parent.start.dim)

        (cp1, p1, n1) = self.copy_skip_subtree(i1)
        p1args = p1.args
        (cp2, p2, n2) = parent.copy_skip_subtree(i2)
        p2args = p2.args

        for i in range(len(p1args)):
            if(p1args[i] == None):
                p1args[i] = n2
                break

        for i in range(len(p2args)):
            if(p2args[i] == None):
                p2args[i] = n1
                break
        
        n1.parent = p2
        n2.parent = p1

        cp1.updateNodeDim()
        cp2.updateNodeDim()

        return (cp1, cp2)
    
    def collapse_and_hoist_mutation(self):
        if(self.start.dim == 1):
            return (self, self)
        i1 = np.random.randint(1, self.start.dim)
        (cp1, p1, n1) = self.copy_skip_subtree(i1)
        p1args = p1.args

        for i in range(len(p1args)):
            if(p1args[i] == None):
                if(np.random.random() < 0.5):#const
                    p1args[i] = Node(const, get_random_element(constPool), p1)
                else: #var
                    p1args[i] = Node(var, np.random.randint(self.nVar), p1)
                break

        cp2 = Expr(self.nVar)
        cp2.start = n1

        cp1.updateNodeDim()
        return (cp1, cp2)
    
    def point_mutation(self):
        if(self.start.dim == 1):
            return self
        i1 = np.random.randint(1, self.start.dim)
        (cp1, p1, n1) = self.copy_skip_subtree(i1)
        p1args = p1.args

        for i in range(len(p1args)):
            if(p1args[i] == None):
                p1args[i] = n1

        n1.parent = p1

        if(n1.op.opType == ops_types.const):
            n1.args.name = None
            n1.args.val += (np.random.random()-0.5)*0.1
        elif(n1.op.opType == ops_types.var):
            n1.args = np.random.randint(self.nVar)
        else:
            n1.op = get_random_element(opsPoolBynArgs[n1.op.nArgs - 1])


        return cp1

    def copy(self):
        cpExp = Expr(self.nVar)

        def _cpNode(node, parent):
            cpN = node.copy()
            cpN.parent = parent
            #cpN.tempRes = None

            args = []

            if(node.op.opType == ops_types.const or node.op.opType == ops_types.var):
                cpN.args = node.args
                return cpN

            for n in node.args:
                args += [_cpNode(n, cpN)]

            cpN.args = args
            return cpN

        cpExp.start = _cpNode(self.start, None)
        return cpExp
    
    def copy_skip_subtree(self, skippedIndex):
        self.skippedNode = None
        self.skippedNodeParent = None
        self.curIndex = 0
        cpExp = Expr(self.nVar)

        def _cpNode(node, parent):
            cpN = node.copy()#Node(node.op, None, parent)
            cpN.parent = parent
            args = []

            #cpN.tempRes = None

            if(node.op.opType == ops_types.const or node.op.opType == ops_types.var):
                cpN.args = node.args
                return cpN

            for n in node.args:
                self.curIndex += 1
                if(self.curIndex == skippedIndex):
                    cpN.tempRes = None
                    self.skippedNodeParent = cpN
                    self.skippedNode = _cpNode(n, None)
                    args += [None]
                else:
                    args += [_cpNode(n, cpN)]
            cpN.args = args

            if((cpN.tempRes is None) and (cpN.parent is not None)):
                cpN.parent.tempRes = None

            return cpN

        cpExp.start = _cpNode(self.start, self.start.parent)
        return (cpExp, self.skippedNodeParent, self.skippedNode)
    
    def print(self):

        def _print(node):
            node.op.print(node.args, _print)

        _print(self.start)
        print('')
    
    def evalFitness(self, x, y):
        mse = 0
        try:
            r = self.getResult(x)
            difs = np.square(r - y)
            mse = difs.mean()
        except:
            mse = np.inf
        
        self.fitness = mse
        return mse

class ExprPopulation:
    def __init__(self, dim, nVar, maxInitD):
        self.population = []
        self.bestExpr = None
        self.bestList = []
        self.nVar = nVar
        self.maxInitD = maxInitD
        self.dim = dim
        self.rand_init(dim)
        

    def rand_init(self, dim):
        for _ in range(dim//2):
            expr = Expr(self.nVar)
            expr.init_random_full(self.maxInitD)
            self.population += [expr]

        dim -= dim//2

        for _ in range(dim):
            expr = Expr(self.maxInitD)
            expr.init_random_grow(self.maxInitD)
            self.population += [expr]

    def select_parent(self):
        a = get_random_element(self.population)
        b = get_random_element(self.population)

        if np.random.random() < 0.1:
            if(a.start.dim < b.start.dim):
                return a
            else:
               return b
        else:
            if(a.fitness < b.fitness):
                return a
            else:
                return b
            
    def extint(self, p):
        a = int(len(self.population)*p)
        self.population = self.population[0 : a]
        #self.rand_init(self.dim - a)

    def evolve(self, nit, chRateo, x, y, showProg = True):
        newPop = []

        for e in self.population:
            r = e.evalFitness(x, y)
            mse = (np.square(r - y)).mean()
            e.fitness = mse

        rng = range(nit)
        if showProg:
            rng = tqdm(rng)

        for _ in rng:

            for _ in range(popDim * chRateo):
                p1 = self.select_parent()
                if(np.random.random() < 0.9): #xover
                    p2 = self.select_parent()
                    (a, b) = p1.exchange_random_subtrees(p2)
                    if(a.evalFitness(x, y) < b.evalFitness(x, y)):
                        newPop += [a]
                    else:
                        newPop += [b]
                else: #mutate
                    (a, b) = p1.collapse_and_hoist_mutation()
                    if(a.evalFitness(x, y) < b.evalFitness(x, y)):
                        newPop += [a]
                    else:
                        newPop += [b]

            #newPop += pop.population
            
            mses = np.zeros(len(newPop))

            k = 0
            for e in newPop:
                mses[k] = e.fitness
                k+=1
            
            sort_idxs = np.array(np.argsort(mses))

            self.population = []
            for k in range(popDim):
                self.population += [newPop[sort_idxs[k]]]

            if((self.bestExpr is None) or self.bestExpr.fitness > self.population[0].fitness):
                self.bestExpr = self.population[0]

            self.bestList += [self.population[0].fitness]
            
            newPop = []

import warnings
warnings.filterwarnings("ignore")

problem = np.load('./data/problem_8.npz')
x = problem['x']
y = problem['y']

popDim = 200
chRateo = 4
nEpoch = 7
nit = 15
nPops = 3

import pickle

'''
with open('2_hi.pkl', 'rb') as inp:
    e = pickle.load(inp)

e.updateConstList()
e.updateNodeDim()

print(e.fitness)
e.print()

exit()
'''
'''
import s333279

res = s333279.f8(x)

print(((res-y)**2).mean())

for i in range(x.shape[0]):
    plt.scatter(x[i,:], y)
    plt.scatter(x[i,:], res)
    plt.show()
'''


print(x.shape)

pops = []

for _ in range(nPops):
    pops += [ExprPopulation(popDim, x.shape[0], 3)]

for ne in range(nEpoch):
    print(ne+1, '/', nEpoch)

    for pop in pops:
        pop.evolve(nit, chRateo, x, y)
        pop.extint(0.1)
    
    mergePop = []
    for k in range(nPops):
        pop = pops[k]
        mergePop += pop.population

    for pop in pops:
        pop.population = list(mergePop)
        pop.rand_init(pop.dim - len(mergePop))

e = None
for pop in pops:
    if((e is None) or pop.bestExpr.fitness < e.fitness):
        e = pop.bestExpr


with open('expr.pkl', 'wb') as outp:
    pickle.dump(e, outp, pickle.HIGHEST_PROTOCOL)

e.updateConstList()

e.print()
print(e.fitness)
e.updateNodeDim()
print(e.start.dim)


for pop in pops:
    plt.scatter(range(len(pop.bestList)), pop.bestList)
plt.show()

for i in range(x.shape[0]):
    plt.scatter(x[i,:], y)
    plt.scatter(x[i,:], e.getResult(x))
    plt.show()


