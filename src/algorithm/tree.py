import pandas as pd
import numpy as np
import json

class IterativeDichotomiser3:
    """
    Iterative Dichotomiser 3 (ID3) decision tree classifier.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    tree_ : dict
        The constructed decision tree represented as a nested dictionary.
    """
    
    def __init__(self):
        self.tree_ = None
        self.attribute_unique_values_ = {}
        
    def fit(self, X, y):
        """
        Fit the classifier with the training data. Construct the decision tree.
        
        Parameters
        ----------
        X : list of shape (n_samples, n_features)
            The training data.
        y : list of shape (n_samples,)
            The target labels corresponding to the training data. Can contain strings or other types.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        
        # Convert the input data to a DataFrame
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X)
            
        # Get the unique values of each categorical attribute
        for attribute in data.columns:
            if data[attribute].dtype == 'object':
                self.attribute_unique_values_[attribute] = data[attribute].unique().tolist()
        
        # Add the target labels to the DataFrame
        if isinstance(y, pd.Series):
            data['target_variable_'] = y
        else:
            data['target_variable_'] = pd.Series(y)
        
        # Construct the decision tree
        self.tree_ = self._construct_tree(data, data.columns[:-1], None)
        
    def predict(self, X):
        """
        Predict the class labels for the input data.
        
        Parameters
        ----------
        X : list of shape (n_samples, n_features)
            The input data to classify.
        
        Returns
        -------
        y_pred : list of shape (n_samples,)
            The predicted class labels for each input sample.
        """
        # Check if the classifier has been fitted
        if self.tree_ is None:
            raise ValueError("The classifier has not been fitted yet.")
        
        # Check if the input data is a DataFrame
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X)
            
        # Make predictions for each sample
        y_pred = data.apply(lambda x: self._classify(x, self.tree_), axis=1)
            
        return y_pred
    
    def _construct_tree(self, examples, attributes, parent_examples):
        """
        Recursively construct the decision tree based on the ID3 algorithm.
        
        Parameters
        ----------
        examples : DataFrame
            The input data for the current node.
        attributes : list
            The list of available attributes to split on.
        parent_examples : DataFrame
            The input data for the parent node.
            
        Returns
        -------
        node : dict
            The decision tree node represented as a nested dictionary.
        """
        # Base case: no samples left
        if len(examples) == 0:
            return self._get_majority_class(parent_examples)
        
        # Base case: all samples have the same class
        if len(examples['target_variable_'].unique()) == 1:
            return examples['target_variable_'].iloc[0]
        
        # Base case: no attributes left to split on
        if len(attributes) == 0:
            return self._get_majority_class(examples)
        
        # Find the best attribute to split on
        best_gain, best_attribute = self._importance(attributes, examples)
        
        # Create a new tree node with the best attribute
        node = {
            'attribute': best_attribute,
            'gain': np.float64(best_gain),
            'majority_class': self._get_majority_class(examples),
            'children': {}
        }
        
        # Recursively construct the tree for each value of the best attribute
        for value in self.attribute_unique_values_[best_attribute]:
            subset = examples.loc[examples[best_attribute] == value].drop(columns=best_attribute)
            
            node['children'][value] = self._construct_tree(subset, attributes.drop(best_attribute), examples)

        return node           
        
    def _get_majority_class(self, examples):
        """
        Determine the majority class of the input samples.
        
        Parameters
        ----------
        examples : DataFrame
            The input data for which to determine the majority class.
        
        Returns
        -------
        majority_class : any
            The majority class label.
        """
        return examples['target_variable_'].value_counts().idxmax()
    
    def _importance(self, attributes, examples):
        """
        Compute the importance of each attribute and find the best attribute to split on.
        
        Parameters
        ----------
        attributes : list
            The list of available attributes to split on.
        examples : DataFrame
            The input data for the current node.
        
        Returns
        -------
        best_attribute : str
            The name of the best attribute to split on.
        best_gain : float
            The computed information gain value.        
        """
        best_attribute = None
        best_gain = -1
        
        # Compute the entropy of the current node
        entropy = self._entropy(examples['target_variable_'])
        
        for attribute in attributes:
            gain = entropy - self._information_gain(examples, attribute)
            if gain > best_gain:
                best_attribute = attribute
                best_gain = gain
    
        return best_gain, best_attribute
    
    def _information_gain(self, examples, attribute):
        """
        Compute the information gain of an attribute.
        
        Parameters
        ----------
        examples : DataFrame
            The input data for the current node.
        attribute : str
            The attribute for which to compute the information gain.
        
        Returns
        -------
        gain : float
            The computed information gain value.
        """
        gain = 0
        for value in examples[attribute].unique():
            subset = examples[examples[attribute] == value]
            gain += len(subset) / len(examples) * self._entropy(subset['target_variable_'])
            
        return gain
    
    def _entropy(self, labels):
        """
        Compute the entropy of a set of labels.
        
        Parameters
        ----------
        labels : Series
            The target labels for which to compute the entropy.
        
        Returns
        -------
        entropy : float
            The computed entropy value.
        """
        value_counts = labels.value_counts(normalize=True)
        return -sum(p * np.log2(p) for p in value_counts)
    
    def _classify(self, x, node):
        """
        Recursively classify a sample using the decision tree.
        
        Parameters
        ----------
        x : list or tuple of shape (n_features,)
            The input sample to classify.
        node : dict
            The current node of the decision tree.
        
        Returns
        -------
        class_label : any
            The predicted class label for the input sample.
        """
        # Base case: leaf node
        if not isinstance(node, dict):
            return node
        
        attribute = node['attribute']
        value = x[attribute]
        
        # Base case: unknown attribute value
        if value not in node['children'].keys():
            return node['majority_class']
        
        return self._classify(x, node['children'][value])
    
    def save(self, filename):
        """
        Save the decision tree to a JSON file.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the decision tree to.
        """
        with open(filename, 'w') as file:
            json.dump(self.tree_, file)
            
    def load(self, filename):
        """
        Load a decision tree from a JSON file.
        
        Parameters
        ----------
        filename : str
            The name of the file to load the decision tree from.
        """
        with open(filename, 'r') as file:
            self.tree_ = json.load(file)