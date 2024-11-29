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
    attribute_unique_values_ : dict
        The unique values of each categorical attribute in the training data.
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
        
        data = pd.DataFrame(X)
        for attribute in data.columns:
            if data[attribute].dtype == 'object' or data[attribute].dtype == 'category':
                self.attribute_unique_values_[attribute] = data[attribute].unique().tolist()
        
        data['target_variable_'] = y
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
        
        data = pd.DataFrame(X)
        y_pred = data.apply(lambda x: self._classify(x, self.tree_), axis=1)
        return y_pred.tolist()

    
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
        best_gain, best_attribute, best_split = self._importance(attributes, examples)
        
        # Create a new tree node with the best attribute
        node = {
            'attribute': best_attribute,
            'gain': np.float64(best_gain),
            'split': best_split,
            'majority_class': self._get_majority_class(examples),
            'children': {}
        }
        
        # Recursively construct the tree for each value of the best attribute
        if best_split is None:
            for value in self.attribute_unique_values_[best_attribute]:
                subset = examples.loc[examples[best_attribute] == value].drop(columns=best_attribute)
                
                node['children'][value] = self._construct_tree(subset, attributes.drop(best_attribute), examples)
        else:
            left_subset = examples.loc[examples[best_attribute] <= best_split].drop(columns=best_attribute)
            right_subset = examples.loc[examples[best_attribute] > best_split].drop(columns=best_attribute)
            
            node['children'][f"<= {best_split}"] = self._construct_tree(left_subset, attributes.drop(best_attribute), examples)
            node['children'][f"> {best_split}"] = self._construct_tree(right_subset, attributes.drop(best_attribute), examples)

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
        return examples['target_variable_'].mode()[0]
    
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
        best_split = None
        
        # Compute the entropy of the current node
        entropy = self._entropy(examples['target_variable_'])
        
        for attribute in attributes:
            # Check if the attribute is categorical or continuous
            if examples[attribute].dtype == 'object' or examples[attribute].dtype == 'category':              
                gain = entropy - self._information_gain(examples, attribute)
                split = None
            else:              
                # Find the best split value for the continuous attribute
                gain, split = self._find_best_split(examples, attribute)
                
            if gain > best_gain:
                best_attribute = attribute
                best_gain = gain
                best_split = split
    
        return best_gain, best_attribute, best_split
    
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
        value_counts = examples[attribute].value_counts(normalize=True)
        gain = sum(value_counts[value] * self._entropy(examples[examples[attribute] == value]['target_variable_']) for value in value_counts.index)
        
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
        return -np.sum(value_counts * np.log2(value_counts))
    
    def _find_best_split(self, examples, attribute):
        """
        Find the best split value for a continuous attribute.
        
        Parameters
        ----------
        examples : DataFrame
            The input data for the current node.
        attribute : str
            The continuous attribute for which to find the best split value.
        
        Returns
        -------
        best_gain : float
            The computed information gain value.
        best_split : float
            The best split value for the continuous attribute.
        """
        best_gain = -1
        best_split = None
        
        # Sort the examples by the attribute value
        examples = examples.sort_values(attribute)
        
        # Compute the initial entropy
        total_entropy = self._entropy(examples['target_variable_'])
        
        # Get the attribute values and target variable
        attribute_values = examples[attribute].values
        target_values = examples['target_variable_']

        # Compute potential splits, where the target variable changes. Compute the index halfway between each pair of values.
        potential_splits = []
        for i in range(1, len(attribute_values)):
            if target_values.values[i] != target_values.values[i - 1]:
                split = (attribute_values[i] + attribute_values[i - 1]) / 2
                potential_splits.append(split)
        
        # Vectorized computation of entropy for each split
        for split in potential_splits:
            left_mask = attribute_values <= split
            right_mask = ~left_mask

            left_entropy = self._entropy(target_values[left_mask])
            right_entropy = self._entropy(target_values[right_mask])

            left_weight = np.sum(left_mask) / len(examples)
            right_weight = np.sum(right_mask) / len(examples)

            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
            gain = total_entropy - weighted_entropy

            if gain > best_gain:
                best_gain = gain
                best_split = split
            
        return best_gain, best_split
    
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
        
        # Check if the attribute is categorical or continuous
        if attribute in self.attribute_unique_values_:
            if value not in node['children']:
                return node['majority_class']
            
            return self._classify(x, node['children'][value])
        else:
            if value <= node['split']:
                return self._classify(x, node['children'][f"<= {node['split']}"])
            else:
                return self._classify(x, node['children'][f"> {node['split']}"])
    
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
        
        self.attribute_unique_values_ = {}
        self._get_unique_values(self.tree_)
        
    def _get_unique_values(self, node):
        """
        Recursively extract the unique values of each categorical attribute in the decision tree.
        
        Parameters
        ----------
        node : dict
            The current node of the decision tree.
        """
        if 'attribute' in node:
            attribute = node['attribute']
            
            if attribute in self.attribute_unique_values_:
                return
            
            # Check if any dictionary in children is numeric by checking if it contains <= and > keys
            if not any('>' in key for key in node['children']):
                self.attribute_unique_values_[attribute] = list(node['children'].keys())
                
            for child in node['children'].values():
                self._get_unique_values(child)