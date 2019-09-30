
'''
NOTE: this is a reduced project file, it does not contain all the steps from the nanodegree lecture
'''

######### parameters
min_count = 100
polarity_cutoff = 0.2


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
print("\tInput is a binary vector < 74074, only containing words that have a certain min count (", min_count, ")")
print("\tand their log positive-negative ratio is at least a certain amount away from 0 ( polarity_cutoff, here:", polarity_cutoff, ")\n")
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=min_count,polarity_cutoff=polarity_cutoff,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])

'''
And run the following cell to test it's performance. It should be around 85 %
'''

print('\n')
print("Test the trained network:")
mlp.test(reviews[-1000:],labels[-1000:])

print('\n\n')
################# End of Project 6.

