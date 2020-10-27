from urllib import request
from os import system, name

repeat = True

'''
Returns the count of times two words appear
in sequence in a given list.
'''
def count_sequence(lst, first, sec):
	count = 0
	indx = 0
	for word in lst:
		if word == first:
			if lst[indx+1] == sec:
				count += 1
		indx += 1
	return count
	
def main():
	while repeat == True:
	
		raw = "learn this learn that learn all you can but also learn to unlearn" 
		tokens = raw.split()
		
		tokensSet = set(tokens)
		print('Length of set: ' + str(len(tokensSet)))
		print('Length of tokens: ' + str(len(tokens)))
		for word in tokens:
			tokensSet.add(word)

		# Input the words to be computed in the bigram
		wordList = input('\nEnter the list of words you want to use to compute your sentence \nprobability. Each should be separated by a space.' +
						' Remember that the program is case sensitive.\n\n').split()

		arr = []

		for i in range(0,len(wordList)):
			t = []
			for j in range(0,len(wordList)):
					#append bigrams to t
					try:
						# Smooth the bigram by applying Laplace smoothing. Add the count
						# of 1 to the numerator and the count of the vocabulary to the denominator.
						t.append((count_sequence(tokens, wordList[i], wordList[j]) + 1) / (tokens.count(wordList[i]) + len(tokensSet)))
					except:
						t.append(0.0)
			arr.append(t)
						
		#Add first row and column to arr
		wordList.insert(0, '#')
		arr.insert(0, wordList)

		for m in range(1,len(wordList)):
			arr[m].insert(0, wordList[m])

		print('\n\n')

		# Print the bigram table
		s = [[str(e) for e in row] for row in arr]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		print('\n'.join(table))

		correctValue = False
		
		# Prompt for sentence input until input sentence matches original list
		while correctValue == False:
			t = True
			nWordList = input('\nEnter the sentence you wish to compute the probability for. It should only be composed of '
													+ 'the words you originally entered.\n').split()
			for word in nWordList:
				if word in wordList:
					continue
				else:
					print('\nOne or more words entered were not contained in the original list. Please try again.\n')
					t = False
					break
			if t == True:
				correctValue = True

		t, r = 0, 1
		answer = 1

		# Compute the final bigram probability
		while r <= len(nWordList)-1:
			word1 = nWordList[t]
			word2 = nWordList[r]
			
			num1 = int(wordList.index(word1))
			num2 = int(wordList.index(word2))

			answer *= arr[num1][num2]
			t += 1
			r += 1

		endProgram = input('\nThe probability of your sentence appearing in the text is exactly ' + str(answer) + '\n\n' +
			  'Would you like to compute another bigram probability? (y\\n)\n')

		if endProgram == "n":
			break

if __name__ == "__main__":
	main()
