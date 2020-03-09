import numpy as np

"""
library for numeric non-linear equation solving

by Gerrit Farren
"""

"""
Solve non-linear equation via recursive relaxation
The function takes four arguments
func 		function to be solved for equivalance to x
accuracy 	desired accuracy
guess 		intial guess
nmax 		maximum number of iterations
"""
def relaxation(func, accuracy=0.0001, guess=1, nmax=10000):
	#define loop counter
	counter = 0
	#arbitrarily set error to somethimng larger than the desired accuracy
	dguess = accuracy*10
	#iterate untill accuaracy is achived or maximum number of iterations is reached
	while (dguess>accuracy and counter <= nmax):
		tmp = guess
		guess = func(guess)
		dguess = np.fabs(tmp-guess)
		counter +=1

	#error if recusion exceeded maximum
	if(counter>nmax):
		raise Exception("Did not converge!")
	return guess

"""
Solve non-linear equation via recursive binary search
The function takes four arguments
func 		function to be solved for equivalance to x
accuracy 	desired accuracy
guesses 	tuple with inital guesses (must be each side of the root)
nmax 		maximum number of iterations
"""
def binary(func, accuracy=0.0001, guesses=(-1,1), nmax=10000):
	
	if (func(guesses[0])*func(guesses[1]) > 0):
		raise Exception("The function needs to have different signs when evaluated at both guesses")
	else:
		#order guesses s.t. func(guess1)<0 and func(guess2)>0
		if(func(guesses[0]) < 0):
			guess1 = guesses[0]
			guess2 = guesses[1]
		else:
			guess2 = guesses[0]
			guess1 = guesses[1]

		#if one of the guesses is already the exact solution return solution
		if(func(guess1) == 0):
				return guess1
		elif(func(guess2) == 0):
			return guess2

		#compute distance between guesses
		distance = np.fabs(guess1 - guess2)
		#define loop counter
		counter = 0
		#iterate untill accuaracy is achived or maximum number of iterations is reached
		while (distance > accuracy and counter <= nmax):

			newGuess = (guess1 + guess2)/2

			if func(newGuess) > 0:
				guess2 = newGuess
			elif func(newGuess) < 0:
				guess1 = newGuess
			else:
				assert(func(newGuess)==0)
				return newGuess	

			distance = np.fabs((guess1 - guess2)/(guess1+guess2)*2)
			counter +=1

		#error if recusion exceeded maximum
		if(counter>nmax):
			raise Exception("Did not converge!")
		return (guess1 + guess2)/2

"""
Solve non-linear equation via recursive secant method
The function takes four arguments
func 		function to be solved for equivalance to x
accuracy 	desired accuracy
guesses 	tuple with inital guesses
nmax 		maximum number of iterations
"""
def secant(func, accuracy=0.001, guesses=(1,2), nmax=10000):
	#arbitrarily set error to somethimng larger than the desired accuracy
	change = 10*accuracy
	#define loop counter
	counter = 0

	guess1, guess2 = guesses
	val1,val2=func(guess1),func(guess2)
	#iterate untill accuaracy is achived or maximum number of iterations is reached
	try:
		while change>accuracy and counter<=nmax:
			guess1, guess2 = guess2,guess2-val2*(guess2-guess1)/(val2-val1)
			val1,val2=val2, func(guess2)
			print(val1, val2, guess1, guess2)
			change=np.fabs(1-val1/val2)#np.fabs(guess2-guess1)#

			counter +=1
	except Exception:
		print("Solver failed after "+str(counter)+" iterations at value "+str(guess2)+".")
	#error if recusion exceeded maximum
	if(counter>nmax):
		raise Exception("Did not converge!")
	return guess2

"""
Solve non-linear equation via recursive secant method
The function takes four arguments
func 		function to be solved for equivalance to x
deriv 		the analytic derivative of the function func
accuracy 	desired accuracy
guess 		intial guess
nmax 		maximum number of iterations
"""
def newton(func, derv, accuracy=0.0001, guess=1, nmax=10000):
	#arbitrarily set error to somethimng larger than the desired accuracy
	change = 10*accuracy
	#define loop counter
	counter = 0

	#iterate untill accuaracy is achived or maximum number of iterations is reached
	while (change>accuracy and counter<=nmax):
		print(func(guess))
		tmp =  guess - func(guess)/derv(guess)
		change=abs(guess-tmp)
		guess=tmp
		counter +=1



	#error if recusion exceeded maximum
	if counter>nmax:
		raise Exception("Did not converge!")
	return guess

