import pandas as pd
import numpy as np
import json

from numba import njit

@njit(parallel=True)
def compute_split_gains(target_values, attribute_values, potential_splits, current_entropy):
    """
    Compute information gains for all potential splits.

    Parameters
    ----------
    target_values : ndarray
        Target variable values corresponding to each example.
    attribute_values : ndarray
        Continuous attribute values for the examples.
    potential_splits : ndarray
        Array of potential split values.
    current_entropy : float
        Entropy of the current node.

    Returns
    -------
    gains : ndarray
        Information gains for each potential split.
    """
    gains = np.zeros(len(potential_splits))

    for i in range(len(potential_splits)):
        split = potential_splits[i]
        
        # Create masks for left and right splits
        left_mask = attribute_values <= split
        right_mask = ~left_mask
        
        # Compute proportions
        proportions_left = left_mask.sum() / len(attribute_values)
        proportions_right = 1 - proportions_left
        
        # Compute unique values and counts for left and right splits
        unique_left, counts_left = numba_unique_with_counts(target_values[left_mask])
        probabilities_left = counts_left / counts_left.sum()
        left_entropy = -np.sum(probabilities_left * np.log2(probabilities_left))
        
        unique_right, counts_right = numba_unique_with_counts(target_values[right_mask])
        probabilities_right = counts_right / counts_right.sum()
        right_entropy = -np.sum(probabilities_right * np.log2(probabilities_right))
        
        # Compute information gain
        gains[i] = current_entropy - (proportions_left * left_entropy + proportions_right * right_entropy)
    
    return gains

@njit
def numba_unique_with_counts(arr):
    """
    A Numba-compatible implementation to compute unique values and their counts.

    Parameters
    ----------
    arr : ndarray
        Input array.

    Returns
    -------
    unique_values : ndarray
        Array of unique values.
    counts : ndarray
        Array of counts corresponding to the unique values.
    """
    unique_values = np.unique(arr)
    counts = np.zeros(len(unique_values), dtype=np.int64)
    for i in range(len(unique_values)):
        counts[i] = np.sum(arr == unique_values[i])
    return unique_values, counts

class IterativeDichotomiser3:
    """
    Iterative Dichotomiser 3 (ID3) decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the decision tree. If None, the tree will expand until all leaves are pure.

    Attributes
    ----------
    tree_ : dict
        The constructed decision tree represented as a nested dictionary.
    classes_ : ndarray
        The unique class labels observed in the training data.
    feature_names_in_ : ndarray
        The names of the features (columns) used in the training data.
    """
    
    def __init__(self, max_depth=None):
        self.tree_ = {}
        self.classes_ = []
        self.feature_names_in_ = []
        self.max_depth_ = max_depth
        
    def fit(self, X, y):
        """
        Fit the classifier to the training data and construct the decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target labels corresponding to the training data.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        
        # If X is not a DataFrame, convert it to one
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store the feature names
        self.feature_names_in_ = np.array(X.columns)
        
        # Store the class labels
        self.classes_ = np.unique(y)
        
        # Encode the target variable
        if isinstance(y, pd.DataFrame):
            y = np.array([np.where(self.classes_ == label)[0][0] for label in y.values.ravel()], dtype=np.int64)
        else:
            y = np.array([np.where(self.classes_ == label)[0][0] for label in y], dtype=np.int64)
        
        # Combine the input data and target variable
        data = X.copy()
        data['target_variable_'] = y
        data = data.values

        # Construct the decision tree        
        self.tree_ = self._construct_tree(data, self.feature_names_in_, None)
        
    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each input sample.
        """
        # Check if the classifier has been fitted
        if self.tree_ is None:
            raise ValueError("The classifier has not been fitted yet.")
        
        # If X is not a DataFrame, convert it to one
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Predict the class labels
        y_pred = X.apply(lambda row: self._classify(row.values, self.tree_), axis=1).tolist()
        
        return np.array(y_pred)

    
    def _construct_tree(self, examples, attributes, parent_examples, current_depth=0):
        """
        Recursively construct the decision tree based on the ID3 algorithm.

        Parameters
        ----------
        examples : ndarray
            The input data for the current node.
        attributes : ndarray
            The list of available attributes to split on.
        parent_examples : ndarray
            The input data for the parent node.
        current_depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        node : dict
            The decision tree node represented as a nested dictionary.
        """
        # Base case: no samples left
        if len(examples) == 0:
            return self._get_majority_class(parent_examples)
        
        # Base case: all samples have the same class
        if len(np.unique(examples[:, -1])) == 1:
            return examples[0, -1]
        
        # Base case: no attributes left to split on or maximum depth reached
        if len(attributes) == 0 or current_depth == self.max_depth_:
            return self._get_majority_class(examples)
                        
        # Find the best attribute to split on
        best_attribute_index, best_split = self._importance(attributes, examples)
        
        # Create a new tree node with the best attribute
        node = {
            'attribute_': attributes[best_attribute_index],
            'split_': best_split,
            'majority_class_': self._get_majority_class(examples),
            'children_': {}
        }
                
        # Remove the best attribute from the list of attributes
        substracted_attributes = np.delete(attributes, best_attribute_index)
        
        # Recursively construct the subtree
        if best_split is None:
            # Get the unique values of the best attribute
            best_attribute_unique_values = np.unique(examples[:, best_attribute_index])
            
            # Create a child node for each unique value of the best attribute
            for value in best_attribute_unique_values:
                # Create subset of examples where the best attribute has the current value
                subset = examples[examples[:, best_attribute_index] == value]      
                
                # Remove the best attribute from the subset
                subset = np.delete(subset, best_attribute_index, axis=1)  
                                
                node['children_'][value] = self._construct_tree(subset, substracted_attributes, examples, current_depth=current_depth+1)
        else:
            # Create a child node for based on the best split value
            left_subset = examples[examples[:, best_attribute_index] <= best_split]
            right_subset = examples[examples[:, best_attribute_index] > best_split]
            
            # Remove the best attribute from the subsets
            left_subset = np.delete(left_subset, best_attribute_index, axis=1)
            right_subset = np.delete(right_subset, best_attribute_index, axis=1)
            
            node['children_'][f"<="] = self._construct_tree(left_subset, substracted_attributes, examples, current_depth=current_depth+1)
            node['children_'][f">"] = self._construct_tree(right_subset, substracted_attributes, examples, current_depth=current_depth+1)

        return node           
        
    def _get_majority_class(self, examples):
        """
        Determine the majority class of the input samples.

        Parameters
        ----------
        examples : ndarray
            The input data for which to determine the majority class.

        Returns
        -------
        majority_class : any
            The majority class label.
        """
        classes, count_class = np.unique(examples[:, -1], return_counts=True)
        return classes[np.argmax(count_class)]
    
    def _importance(self, attributes, examples):
        """
        Compute the importance of each attribute and find the best attribute to split on.

        Parameters
        ----------
        attributes : ndarray
            The list of available attributes to split on.
        examples : ndarray
            The input data for the current node.

        Returns
        -------
        best_attribute : int
            The index of the best attribute to split on.
        best_split : float or None
            The best split value for a continuous attribute, or None for categorical attributes.
        """
        best_attribute = None
        best_gain = -1
        best_split = None
        
        # Compute the entropy of the current node
        entropy = self._entropy(examples[:, -1])
        
        # Find the best attribute to split on
        for i in range(len(attributes)):            
            # Check if the type is 'str' or similar
            if type(examples[0, i]) == str or type(examples[0, i]) == np.str_:
                gain = self._information_gain(examples, i, entropy)
                split = None
            else:
                gain, split = self._find_best_split(examples, i, entropy)
                            
            if gain > best_gain:
                best_attribute = i
                best_gain = gain
                best_split = split
                
        return best_attribute, best_split
    
    def _information_gain(self, examples, attribute, entropy):
        """
        Compute the information gain of a categorical attribute.

        Parameters
        ----------
        examples : ndarray
            The input data for the current node.
        attribute : int
            The index of the attribute for which to compute the information gain.
        entropy : float
            The entropy of the current node

        Returns
        -------
        gain : float
            The computed information gain value.
        """
        # Initialize the gain
        gain = entropy
        
        # Compute proportions of each unique value of the attribute
        values, count_values = np.unique(examples[:, attribute], return_counts=True)
        proportions = count_values / len(examples)
        
        # Compute the entropy of each subset of the attribute
        for i in range(len(values)):
            subset = examples[examples[:, attribute] == values[i]]
            subset_entropy = self._entropy(subset[:, -1])
            
            gain -= proportions[i] * subset_entropy      
        
        return gain
    
    def _entropy(self, examples_class):
        """
        Compute the entropy of a set of examples.

        Parameters
        ----------
        examples_class : ndarray
            The target variable of the input data.

        Returns
        -------
        entropy : float
            The computed entropy value.
        """
        # Compute the proportion of each class
        classes, count_classes = np.unique(examples_class, return_counts=True)
        proportions = count_classes / len(examples_class)
        entropy = -np.sum(proportions * np.log2(proportions))
        
        return entropy
    
    def _find_best_split(self, examples, attribute, entropy):
        """
        Find the best split value for a continuous attribute.

        Parameters
        ----------
        examples : ndarray
            The input data for the current node.
        attribute : int
            The index of the continuous attribute to split on.
        entropy : float
            The entropy of the current node.

        Returns
        -------
        best_gain : float
            The computed information gain value.
        best_split : float
            The best split value for the continuous attribute.
        """        
        best_gain = 0
        best_split = None
        
        # Sort the examples by the attribute value
        examples_sorted = examples[np.argsort(examples[:, attribute])]
        
        # Get the attribute values and target variable
        attribute_values = examples_sorted[:, attribute]
        target_values = examples_sorted[:, -1]

        # Compute potential splits, where the target variable changes. Compute the index halfway between each pair of values.
        change_indices = np.where(target_values[:-1] != target_values[1:])[0]
        
        # Check if there are no changes in the target variable
        if len(change_indices) == 0:
            return best_gain, best_split
        
        # Compute the potential split values
        potential_splits = (attribute_values[change_indices] + attribute_values[change_indices + 1]) / 2
        
        # Don't check the same split value twice
        potential_splits = np.unique(potential_splits)
                
        # Set attribute values
        attribute_values = attribute_values.astype(np.float64)
        target_values = target_values.astype(np.int64)
        potential_splits = potential_splits.astype(np.float64)
        
        # Compute information gains for all potential splits
        gains = compute_split_gains(target_values, attribute_values, potential_splits, entropy)
        
        # Get the best split value
        best_split = potential_splits[np.argmax(gains)]
        best_gain = np.max(gains)
        
        return best_gain, best_split
    
    def _classify(self, x, node):
        """
        Recursively classify a sample using the decision tree.

        Parameters
        ----------
        x : array-like of shape (n_features,)
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
            return self.classes_[node]
        
        attribute = node['attribute_']
        attribute_index = np.where(self.feature_names_in_ == attribute)[0][0]
        value = x[attribute_index]
        
        # Check if the attribute is categorical or numerical
        if node['split_'] is None:
            if value not in node['children_']:
                return node['majority_class_']
            
            return self._classify(x, node['children_'][value])
        else:
            if value <= node['split_']:
                return self._classify(x, node['children_']["<="])
            else:
                return self._classify(x, node['children_'][">"])
    
    def save(self, filename):
        """
        Save the decision tree to a JSON file.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the decision tree to.
        """
        decision_tree = {
            'tree': self.tree_,
            'classes': self.classes_.tolist(),
            'feature_names_in': self.feature_names_in_.tolist()
        }
        
        with open(filename, 'w') as file:
            json.dump(decision_tree, file)
            
    def load(self, filename):
        """
        Load a decision tree from a JSON file.
        
        Parameters
        ----------
        filename : str
            The name of the file to load the decision tree from.
        """
        with open(filename, 'r') as file:
            decision_tree = json.load(file)
        
        self.tree_ = decision_tree['tree']
        self.classes_ = np.array(decision_tree['classes'])
        self.feature_names_in_ = np.array(decision_tree['feature_names_in'])