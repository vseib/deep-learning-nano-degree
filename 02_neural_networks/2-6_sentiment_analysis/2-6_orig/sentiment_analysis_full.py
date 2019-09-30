
'''
NOTE: this is a reduced project file, it does not contain all the steps from the nanodegree lecture
'''

####### Lesson: Curate a Dataset

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('../data/reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('../data/labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

'''
Note: The data in reviews.txt we're using has already been preprocessed a bit and contains only lower case characters. If we were working from raw data, where we didn't know it was all lower case, we would want to add a step here to convert it. That's so we treat different variations of the same word, like The, the, and THE, all the same way.
'''

print("number of reviews in dataset: ", len(reviews))


################################################
############# Project 1: Quick Theory Validation
################################################

from collections import Counter
import numpy as np

# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

'''
TODO: Examine all the reviews. For each word in a positive review, increase the count for that word in both your positive counter and the total words counter; likewise, for each word in a negative review, increase the count for that word in both your negative counter and the total words counter.

Note: Throughout these projects, you should use split(' ') to divide a piece of text (such as a review) into individual words. If you use split() instead, you'll get slightly different results than what the videos and solutions show.
'''

# TODO: Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
for i in range(len(reviews)):
    for word in reviews[i].split(' '):
        total_counts.update([word])
        if labels[i] == 'POSITIVE':
            positive_counts.update([word])
        if labels[i] == 'NEGATIVE':
            negative_counts.update([word])
        
# Examine the counts of the most common words in positive reviews
print("\nMost common words in positive reviews: \n", list(positive_counts.most_common())[0:20])

# Examine the counts of the most common words in negative reviews
print("\nMost common words in negative reviews: \n", list(negative_counts.most_common())[0:20])

'''
As you can see, common words like "the" appear very often in both positive and negative reviews. Instead of finding the most common words in positive or negative reviews, what you really want are the words found in positive reviews more often than in negative reviews, and vice versa. To accomplish this, you'll need to calculate the ratios of word usage between positive and negative reviews.

TODO: Check all the words you've seen and calculate the ratio of postive to negative uses and store that ratio in pos_neg_ratios. 

    Hint: the positive-to-negative ratio for a given word can be calculated with
    positive_counts[word] / float(negative_counts[word]+1). Notice the +1 in the denominator – that
    ensures we don't divide by zero for words that are only seen in positive reviews.
'''

# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times
for word, number in total_counts.items():
    if number > 100:
        ratio = positive_counts[word] / float(negative_counts[word]+1)
        pos_neg_ratios.update({word: ratio})

'''
Examine the ratios you've calculated for a few words:
'''

print('\n')
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

'''
Looking closely at the values you just calculated, we see the following:

    Words that you would expect to see more often in positive reviews – like "amazing" – have a ratio greater than 1. The more skewed a word is toward postive, the farther from 1 its positive-to-negative ratio will be.
    Words that you would expect to see more often in negative reviews – like "terrible" – have positive values that are less than 1. The more skewed a word is toward negative, the closer to zero its positive-to-negative ratio will be.
    Neutral words, which don't really convey any sentiment because you would expect to see them in all sorts of reviews – like "the" – have values very close to 1. A perfectly neutral word – one that was used in exactly the same number of positive reviews as negative reviews – would be almost exactly 1. The +1 we suggested you add to the denominator slightly biases words toward negative, but it won't matter because it will be a tiny bias and later we'll be ignoring words that are too close to neutral anyway.

Ok, the ratios tell us which words are used more often in postive or negative reviews, but the specific values we've calculated are a bit difficult to work with. A very positive word like "amazing" has a value above 4, whereas a very negative word like "terrible" has a value around 0.18. Those values aren't easy to compare for a couple of reasons:

    Right now, 1 is considered neutral, but the absolute value of the postive-to-negative rations of very postive words is larger than the absolute value of the ratios for the very negative words. So there is no way to directly compare two numbers and see if one word conveys the same magnitude of positive sentiment as another word conveys negative sentiment. So we should center all the values around netural so the absolute value from neutral of the postive-to-negative ratio for a word would indicate how much sentiment (positive or negative) that word conveys.
    When comparing absolute values it's easier to do that around zero than one.

To fix these issues, we'll convert all of our ratios to new values using logarithms.

TODO: Go through all the ratios you calculated and convert them to logarithms. (i.e. use np.log(ratio))

In the end, extremely positive and extremely negative words will have positive-to-negative ratios with similar magnitudes but opposite signs.
'''

# TODO: Convert ratios to logs
pos_neg_ratios = Counter()

for word, number in total_counts.items():
    if number > 100:
        ratio = positive_counts[word] / float(negative_counts[word]+1)
        if(ratio > 1):
            pos_neg_ratios[word] = np.log(ratio)
        else:
            pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

'''
NOTE: In the video, Andrew uses the following formulas for the previous cell:

        For any postive words, convert the ratio using np.log(ratio)
        For any negative words, convert the ratio using -np.log(1/(ratio + 0.01))

These won't give you the exact same results as the simpler code we show in this notebook, but the values will be similar. In case that second equation looks strange, here's what it's doing: First, it divides one by a very small number, which will produce a larger positive number. Then, it takes the log of that, which produces numbers similar to the ones for the postive words. Finally, it negates the values by adding that minus sign up front. The results are extremely positive and extremely negative words having positive-to-negative ratios with similar magnitudes but oppositite signs, just like when we use np.log(ratio).

Examine the new ratios you've calculated for the same words from before:

'''

print("Log Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Log Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Log Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

'''
If everything worked, now you should see neutral words with values close to zero. In this case, "the" is near zero but slightly positive, so it was probably used in more positive reviews than negative reviews. But look at "amazing"'s ratio - it's above 1, showing it is clearly a word with positive sentiment. And "terrible" has a similar score, but in the opposite direction, so it's below -1. It's now clear that both of these words are associated with specific, opposing sentiments.

Now run the following cells to see more ratios.

The first cell displays all the words, ordered by how associated they are with postive reviews. (Your notebook will most likely truncate the output so you won't actually see all the words in the list.)

The second cell displays the 30 words most associated with negative reviews by reversing the order of the first list and then looking at the first 30 words. (If you want the second cell to display all the words, ordered by how associated they are with negative reviews, you could just write reversed(pos_neg_ratios.most_common()).)

You should continue to see values similar to the earlier ones we checked – neutral words will be close to 0, words will get more positive as their ratios approach and go above 1, and words will get more negative as their ratios approach and go below -1. That's why we decided to use the logs instead of the raw ratios.
'''

# words most frequently seen in a review with a "POSITIVE" label
print("\nMost common words in positive reviews: ", list(pos_neg_ratios.most_common())[0:20])

# words most frequently seen in a review with a "NEGATIVE" label
print("\nMost common words in negative reviews: ", list(reversed(pos_neg_ratios.most_common()))[0:20])

# Note: Above is the code Andrew uses in his solution video, 
#       so we've included it here to avoid confusion.
#       If you explore the documentation for the Counter class, 
#       you will see you could also find the 30 least common
#       words like this: pos_neg_ratios.most_common()[:-31:-1]


################# End of Project 1

################# Transforming Text into Numbers


################################################
############# Project 2: Creating the Input/Output Data
################################################

'''
Project 2 is omitted here as it will be contained later inside the methods of the class.
This project shows how to create a dataset. First we count the number of all different words in the
reviews (74074). Then we create an input vector of that size and we will transform each 
review to such a vector as input to the neural network.
An auxiliary function that we will create is mapping each word to the index in that input vector.
'''
############### End of Project 2.



################################################
############# Project 3: Building a Neural Network
################################################

'''
TODO: We've included the framework of a class called SentimentNetork. Implement all of the items marked TODO in the code. These include doing the following:

    Create a basic neural network much like the networks you've seen in earlier lessons and in Project 1, with an input layer, a hidden layer, and an output layer.
    Do not add a non-linearity in the hidden layer. That is, do not use an activation function when calculating the hidden layer outputs.
    Re-use the code from earlier in this notebook to create the training data (see TODOs in the code)
    Implement the pre_process_data function to create the vocabulary for our training data generating functions
    Ensure train trains over the entire corpus
'''

import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        for rev in reviews:
            for word in rev.split(' '):
                review_vocab.add(word)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for l in labels:
            label_vocab.add(l)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for idx, word in enumerate(self.review_vocab):
            self.word2index[word] = idx
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for idx, label in enumerate(self.label_vocab):
            self.label2index[label] = idx
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = None
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = None
        sigma = self.hidden_nodes**-0.5 # NOTE: recommended in the course
        self.weights_1_2 = np.random.normal(0, sigma, (self.hidden_nodes, self.output_nodes))
        
        # TODO: Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):
        # TODO: You can copy most of the code you wrote for update_input_layer 
        #       earlier in this notebook. 
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
        
        # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
        # count how many times each word is used in the given review and store the results in layer_0 
        for word in review.split(' '):
            idx = self.word2index.get(word, -1)
            if idx != -1:
                self.layer_0[0][idx] += 1
                
    def get_target_for_label(self,label):
        # TODO: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        if label == "POSITIVE":
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # TODO: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # TODO: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # TODO: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            self.update_input_layer(review)
            layer_1_in = np.dot(self.layer_0, self.weights_0_1)
        
            layer_1_out = layer_1_in
            layer_2_in = np.dot(layer_1_out, self.weights_1_2)
            layer_2_out = self.sigmoid(layer_2_in)
            
            # TODO: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.
          
            ####### calc errors
            layer_2_error = layer_2_out - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2_out)
            layer_1_error = np.dot(layer_2_delta, self.weights_1_2.T)
            layer_1_delta = layer_1_error * 1
            ####### calc weight updates
            self.weights_1_2 -= self.learning_rate * np.dot(layer_1_out.T, layer_2_delta)
            self.weights_0_1 -= self.learning_rate * np.dot(self.layer_0.T, layer_1_delta)       
            
            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            # Keep track of correct predictions.
            if(layer_2_out >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2_out < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        review_processed = review.lower()
        self.update_input_layer(review_processed)
        hidden_input = np.dot(self.layer_0, self.weights_0_1)
        hidden_output = hidden_input
        input_to_output_layer = np.dot(hidden_output, self.weights_1_2)
        output = self.sigmoid(input_to_output_layer)
        
        # TODO: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        if output >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

'''
Run the following cell to create a SentimentNetwork that will train on all but the last 1000 reviews (we're saving those for testing). Here we use a learning rate of 0.1.
'''

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)

'''
Run the following cell to test the network's performance against the last 1000 reviews (the ones we held out from our training set).

We have not trained the model yet, so the results should be about 50% as it will just be guessing and there are only two possible values to choose from.
'''

print('\n')
print("Test the untrained network (random guessing):")
mlp.test(reviews[-1000:],labels[-1000:])
print('\n')

'''
Run the following cell to actually train the network. During training, it will display the model's accuracy repeatedly as it trains so you can see how well it's doing.
'''

print("Training 1st version")
print("\tInput is a vector of size 74074, each element is a count of how many")
print("\ttimes a word from the vocabulary appeared in the review\n")
mlp.train(reviews[:-1000],labels[:-1000])

############### End of Project 3.


################################################
############# Project 4: Reducing Noise in Our Input Data
################################################

'''
TODO: Attempt to reduce the noise in the input data like Andrew did in the previous video. Specifically, do the following:

    Copy the SentimentNetwork class you created earlier into the following cell.
    Modify update_input_layer so it does not count how many times each word is used, but rather just stores whether or not a word was used.
'''

############# NOTE: the only change is in the function: update_input_layer
############# before: self.layer_0[0][idx] = +1
############# now:    self.layer_0[0][idx] = 1
############# effect: don't count word occurances, but make them binary

# TODO: -Copy the SentimentNetwork class from Projet 3 lesson
#       -Modify it to reduce noise, like in the video 
import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        for rev in reviews:
            for word in rev.split(' '):
                review_vocab.add(word)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for l in labels:
            label_vocab.add(l)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for idx, word in enumerate(self.review_vocab):
            self.word2index[word] = idx
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for idx, label in enumerate(self.label_vocab):
            self.label2index[label] = idx
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = None
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = None
        sigma = self.hidden_nodes**-0.5 # NOTE: recommended in the course
        self.weights_1_2 = np.random.normal(0, sigma, (self.hidden_nodes, self.output_nodes))
        
        # TODO: Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):
        # TODO: You can copy most of the code you wrote for update_input_layer 
        #       earlier in this notebook. 
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
        
        # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
        # count how many times each word is used in the given review and store the results in layer_0 
        for word in review.split(' '):
            idx = self.word2index.get(word, -1)
            if idx != -1:
                self.layer_0[0][idx] = 1
                
    def get_target_for_label(self,label):
        # TODO: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        if label == "POSITIVE":
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # TODO: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # TODO: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # TODO: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            self.update_input_layer(review)
            layer_1_in = np.dot(self.layer_0, self.weights_0_1)
        
            layer_1_out = layer_1_in
            layer_2_in = np.dot(layer_1_out, self.weights_1_2)
            layer_2_out = self.sigmoid(layer_2_in)
            
            # TODO: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.
          
            ####### calc errors
            layer_2_error = layer_2_out - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2_out)
            layer_1_error = np.dot(layer_2_delta, self.weights_1_2.T)
            layer_1_delta = layer_1_error * 1
            ####### calc weight updates
            self.weights_1_2 -= self.learning_rate * np.dot(layer_1_out.T, layer_2_delta)
            self.weights_0_1 -= self.learning_rate * np.dot(self.layer_0.T, layer_1_delta)       
            
            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            # Keep track of correct predictions.
            if(layer_2_out >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2_out < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        review_processed = review.lower()
        self.update_input_layer(review_processed)
        hidden_input = np.dot(self.layer_0, self.weights_0_1)
        hidden_output = hidden_input
        input_to_output_layer = np.dot(hidden_output, self.weights_1_2)
        output = self.sigmoid(input_to_output_layer)
        
        # TODO: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        if output >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

'''
Run the following cell to recreate the network and train it. Notice we've gone back to the higher learning rate of 0.1.
'''

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)

print('\n\n')
print("Training 2nd version")
print("\tInput is a vector of size 74074, each element is 0 or 1 and indicates")
print("\twhether or not a word from the vocabulary appeared in the review\n")
mlp.train(reviews[:-1000],labels[:-1000])

'''
That should have trained much better than the earlier attempts. It's still not wonderful, but it should have improved dramatically. Run the following cell to test your model with 1000 predictions.
'''

print('\n')
print("Test the trained network:")
mlp.test(reviews[-1000:],labels[-1000:])

############### End of Project 4.


################################################
############# Project 5: Making our Network More Efficient 
################################################

'''
TODO: Make the SentimentNetwork class more efficient by eliminating unnecessary multiplications and additions that occur during forward and backward propagation. To do that, you can do the following:

    Copy the SentimentNetwork class from the previous project into the following cell.
    Remove the update_input_layer function - you will not need it in this version.
    Modify init_network:

            You no longer need a separate input layer, so remove any mention of self.layer_0
            You will be dealing with the old hidden layer more directly, so create self.layer_1, a two-dimensional matrix with shape 1 x hidden_nodes, with all values initialized to zero

    Modify train:

            Change the name of the input parameter training_reviews to training_reviews_raw. This will help with the next step.
            At the beginning of the function, you'll want to preprocess your reviews to convert them to a list of indices (from word2index) that are actually used in the review. This is equivalent to what you saw in the video when Andrew set specific indices to 1. Your code should create a local list variable named training_reviews that should contain a list for each review in training_reviews_raw. Those lists should contain the indices for words found in the review.
            Remove call to update_input_layer
            Use self's layer_1 instead of a local layer_1 object.
            In the forward pass, replace the code that updates layer_1 with new logic that only adds the weights for the indices used in the review.
            When updating weights_0_1, only update the individual weights that were used in the forward pass.

    Modify run:

            Remove call to update_input_layer
            Use self's layer_1 instead of a local layer_1 object.
            Much like you did in train, you will need to pre-process the review so you can work with word indices, then update layer_1 by adding weights for the indices used in the review.

'''

# TODO: -Copy the SentimentNetwork class from Project 4 lesson
#       -Modify it according to the above instructions 

import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        for rev in reviews:
            for word in rev.split(' '):
                review_vocab.add(word)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for l in labels:
            label_vocab.add(l)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for idx, word in enumerate(self.review_vocab):
            self.word2index[word] = idx
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for idx, label in enumerate(self.label_vocab):
            self.label2index[label] = idx
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_0_1 = None
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        self.weights_1_2 = None
        sigma = self.hidden_nodes**-0.5 # NOTE: recommended in the course
        self.weights_1_2 = np.random.normal(0, sigma, (self.hidden_nodes, self.output_nodes))
        
        # TODO: You no longer need a separate input layer, so remove any mention of self.layer_0
        # TODO: You will be dealing with the old hidden layer more directly,so create self.layer_1,
        #       a two-dimensional matrix with shape 1 x hidden_nodes, with all values initialized to zero
        self.layer_1 = np.zeros((1, hidden_nodes))
                
    def get_target_for_label(self,label):
        if label == "POSITIVE":
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    # TODO Change the name of the input parameter training_reviews to training_reviews_raw.
    # This will help with the next step.
    def train(self, training_reviews_raw, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews_raw) == len(training_labels))

        # TODO At the beginning of the function, you'll want to preprocess your reviews
        # to convert them to a list of indices (from word2index) that are actually used
        # in the review. This is equivalent to what you saw in the video when Andrew set
        # specific indices to 1. Your code should create a local list variable named
        # training_reviews that should contain a list for each review in training_reviews_raw.
        # Those lists should contain the indices for words found in the review.

        training_reviews = list()
        for review_raw in training_reviews_raw:
            indices = set()
            for word in review_raw.split(' '):
                idx = self.word2index.get(word, -1)
                if idx != -1:
                    indices.add(idx) # ---> contains the indices of the input vector that are 1
            training_reviews.append(indices)      
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            # TODO Remove call to update_input_layer
            # TODO Use self's layer_1 instead of a local layer_1 object.
            # TODO In the forward pass, replace the code that updates layer_1
            #      with new logic that only adds the weights for the indices used in the review.
            self.layer_1 *= 0
            for idx in review:
                self.layer_1 += self.weights_0_1[idx]
        
            layer_1_out = self.layer_1 # identical output, no sigmoid
            layer_2_in = np.dot(layer_1_out, self.weights_1_2)
            layer_2_out = self.sigmoid(layer_2_in)      
          
            ####### calc errors
            layer_2_error = layer_2_out - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2_out)
            layer_1_error = np.dot(layer_2_delta, self.weights_1_2.T)
            layer_1_delta = layer_1_error * 1
            ####### calc weight updates
            self.weights_1_2 -= self.learning_rate * np.dot(layer_1_out.T, layer_2_delta)
            # TODO When updating weights_0_1, only update the individual weights that were used in the forward pass.
            for idx in review:
                self.weights_0_1[idx] -= self.learning_rate * layer_1_delta[0]
                       
            if(layer_2_out >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2_out < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """        

        review_processed = review.lower()
        # TODO Remove call to update_input_layer
        # TODO Use self's layer_1 instead of a local layer_1 object.
        # TODO Much like you did in train, you will need to pre-process the review
        # so you can work with word indices, then update layer_1 by adding weights
        # for the indices used in the review.
        indices = set()
        for word in review_processed.split(' '):
            idx = self.word2index.get(word, -1)
            if idx != -1:
                indices.add(idx)
        indices = list(indices)
        
        self.layer_1 *= 0
        for idx in indices:
            self.layer_1 += self.weights_0_1[idx]

        layer_1_out = self.layer_1 # identical output, no sigmoid
        layer_2_in = np.dot(layer_1_out, self.weights_1_2)
        layer_2_out = self.sigmoid(layer_2_in)  
        
        if layer_2_out >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

'''
Run the following cell to recreate the network and train it once again.
'''

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)

print('\n\n')
print("Training 3rd version (faster than 2nd version)")
print("\tInput is a vector of size 74074, each element is 0 or 1 and indicates")
print("\twhether or not a word from the vocabulary appeared in the review\n")
mlp.train(reviews[:-1000],labels[:-1000])

'''
That should have trained much better than the earlier attempts. Run the following cell to test your model with 1000 predictions.
'''

print('\n')
print("Test the trained network:")
mlp.test(reviews[-1000:],labels[-1000:])

############### End of Project 5.

################################################
############# Project 6: Reducing Noise by Strategically Reducing the Vocabulary
################################################

'''
TODO: Improve SentimentNetwork's performance by reducing more noise in the vocabulary. Specifically, do the following:

    Copy the SentimentNetwork class from the previous project into the following cell.
    Modify pre_process_data:

            Add two additional parameters: min_count and polarity_cutoff
            Calculate the positive-to-negative ratios of words used in the reviews. (You can use code you've written elsewhere in the notebook, but we are moving it into the class like we did with other helper code earlier.)
            Andrew's solution only calculates a postive-to-negative ratio for words that occur at least 50 times. This keeps the network from attributing too much sentiment to rarer words. You can choose to add this to your solution if you would like.
            Change so words are only added to the vocabulary if they occur in the vocabulary more than min_count times.
            Change so words are only added to the vocabulary if the absolute value of their postive-to-negative ratio is at least polarity_cutoff

    Modify __init__:

            Add the same two parameters (min_count and polarity_cutoff) and use them when you call pre_process_data
'''


# TODO: -Copy the SentimentNetwork class from Project 5 lesson
#       -Modify it according to the above instructions 
import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    # TODO Add the same two parameters
    #      (min_count and polarity_cutoff) and use them when you call pre_process_data
    def __init__(self, reviews, labels, min_count, polarity_cutoff, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            min_count(int) - Words are only added to the vocabulary if they occur more than min_count times
            polarity_cutoff(float) - Words are only added to the vocabulary if the absolute value of their 
                                     postive-to-negative ratio is at least polarity_cutoff
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    # TODO Add two additional parameters: min_count and polarity_cutoff
    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):
        
        # TODO Calculate the positive-to-negative ratios of words used in the reviews.
        #      (You can use code you've written elsewhere in the notebook, but we are
        #      moving it into the class like we did with other helper code earlier.)


        #### NOTE: using these objects from above outside the class
        '''
        from collections import Counter
        # Create three Counter objects to store positive, negative and total counts
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()
        # Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
        for i in range(len(reviews)):
            for word in reviews[i].split(' '):
                total_counts.update([word])
                if labels[i] == 'POSITIVE':
                    positive_counts.update([word])
                if labels[i] == 'NEGATIVE':
                    negative_counts.update([word])

        # TODO Andrew's solution only calculates a postive-to-negative ratio for words that
        #      occur at least 50 times. This keeps the network from attributing too much
        #      sentiment to rarer words. You can choose to add this to your solution if you would like. 
        # Calculate the ratios of positive and negative uses of the most common words
        # Convert ratios to logs
        pos_neg_ratios = Counter()
        for word, number in total_counts.items():
            if number > min_count:
                ratio = positive_counts[word] / float(negative_counts[word]+1)
                pos_neg_ratios.update({word: ratio})
     
        for word, number in pos_neg_ratios.items():
            if number > 1:
                pos_neg_ratios[word] = np.log(number)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        '''
                
        # TODO Change so words are only added to the vocabulary if they occur in
        #      the vocabulary more than min_count times.
        # TODO Change so words are only added to the vocabulary if the absolute value
        #      of their postive-to-negative ratio is at least polarity_cutoff
        review_vocab = set()
        for rev in reviews:
            for word in rev.split(' '):
                if total_counts[word] > min_count and np.abs(pos_neg_ratios[word]) >= polarity_cutoff:
                    review_vocab.add(word)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        
        label_vocab = set()
        for l in labels:
            label_vocab.add(l)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for idx, word in enumerate(self.review_vocab):
            self.word2index[word] = idx
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for idx, label in enumerate(self.label_vocab):
            self.label2index[label] = idx
        
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_0_1 = None
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        self.weights_1_2 = None
        sigma = self.hidden_nodes**-0.5 # NOTE: recommended in the course
        self.weights_1_2 = np.random.normal(0, sigma, (self.hidden_nodes, self.output_nodes))
        
        # TODO: You no longer need a separate input layer, so remove any mention of self.layer_0
        # TODO: You will be dealing with the old hidden layer more directly,so create self.layer_1,
        #       a two-dimensional matrix with shape 1 x hidden_nodes, with all values initialized to zero
        self.layer_1 = np.zeros((1, hidden_nodes))
                
    def get_target_for_label(self,label):
        if label == "POSITIVE":
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    # TODO Change the name of the input parameter training_reviews to training_reviews_raw.
    # This will help with the next step.
    def train(self, training_reviews_raw, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews_raw) == len(training_labels))

        # TODO At the beginning of the function, you'll want to preprocess your reviews
        # to convert them to a list of indices (from word2index) that are actually used
        # in the review. This is equivalent to what you saw in the video when Andrew set
        # specific indices to 1. Your code should create a local list variable named
        # training_reviews that should contain a list for each review in training_reviews_raw.
        # Those lists should contain the indices for words found in the review.

        training_reviews = list()
        for review_raw in training_reviews_raw:
            indices = set()
            for word in review_raw.split(' '):
                idx = self.word2index.get(word, -1)
                if idx != -1:
                    indices.add(idx) # ---> contains the indices of the input vector that are 1
            training_reviews.append(indices)      
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            # TODO Remove call to update_input_layer
            # TODO Use self's layer_1 instead of a local layer_1 object.
            # TODO In the forward pass, replace the code that updates layer_1
            #      with new logic that only adds the weights for the indices used in the review.
            self.layer_1 *= 0
            for idx in review:
                self.layer_1 += self.weights_0_1[idx]
        
            layer_1_out = self.layer_1 # identical output, no sigmoid
            layer_2_in = np.dot(layer_1_out, self.weights_1_2)
            layer_2_out = self.sigmoid(layer_2_in)      
          
            ####### calc errors
            layer_2_error = layer_2_out - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2_out)
            layer_1_error = np.dot(layer_2_delta, self.weights_1_2.T)
            layer_1_delta = layer_1_error * 1
            ####### calc weight updates
            self.weights_1_2 -= self.learning_rate * np.dot(layer_1_out.T, layer_2_delta)
            # TODO When updating weights_0_1, only update the individual weights that were used in the forward pass.
            for idx in review:
                self.weights_0_1[idx] -= self.learning_rate * layer_1_delta[0]
                       
            if(layer_2_out >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2_out < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """        

        review_processed = review.lower()
        # TODO Remove call to update_input_layer
        # TODO Use self's layer_1 instead of a local layer_1 object.
        # TODO Much like you did in train, you will need to pre-process the review
        # so you can work with word indices, then update layer_1 by adding weights
        # for the indices used in the review.
        indices = set()
        for word in review_processed.split(' '):
            idx = self.word2index.get(word, -1)
            if idx != -1:
                indices.add(idx)
        indices = list(indices)
        
        self.layer_1 *= 0
        for idx in indices:
            self.layer_1 += self.weights_0_1[idx]

        layer_1_out = self.layer_1 # identical output, no sigmoid
        layer_2_in = np.dot(layer_1_out, self.weights_1_2)
        layer_2_out = self.sigmoid(layer_2_in)  
        
        if layer_2_out >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

'''
Run the following cell to train your network with a small polarity cutoff.
'''

'''
That should have trained much better than the earlier attempts. Run the following cell to test your model with 1000 predictions.
'''

print('\n\n')
print("Training 4th version (the greater the polarity_cutoff, the more we sacrifice accuracy for speed)")
print("\tInput is a binary vector < 74074, only containing words that have a certain min count (20)")
print("\tand their log positive-negative ratio is at least a certain amount away from 0 (polarity_cutoff, here: 0.05)\n")
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])

'''
And run the following cell to test it's performance. It should be around 85 %
'''

print('\n')
print("Test the trained network:")
mlp.test(reviews[-1000:],labels[-1000:])

'''
Run the following cell to train your network with a much larger polarity cutoff.
'''

print('\n\n')
print("Training 4th version (the greater the polarity_cutoff, the more we sacrifice accuracy for speed)")
print("\tInput is a binary vector < 74074, only containing words that have a certain min count (20)")
print("\tand their log positive-negative ratio is at least a certain amount away from 0 (polarity_cutoff, here: 0.8)\n")
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])

'''
And run the following cell to test it's performance. It should be around 82 % - we sacrificed some accuracy for a major speed-up in training.
'''
print('\n')
print("Test the trained network:")
mlp.test(reviews[-1000:],labels[-1000:])
print('\n\n')
################# End of Project 6.

