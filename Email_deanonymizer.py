import nltk
import random
import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV - Olivier Grisel <olivier.grisel@ensta.org>


	# Set dataset_location to the location of your CLEANED files
	dataset_location = "F:/Machine Learning/enron_mail_20150507.tar/sent mail"

	# Load the dataset and check that it loaded properly, I had 57606 samples
	dataset = load_files(dataset_location, shuffle=False)
	print("n_samples: %d" % len(dataset.data))

	# Split the dataset, and check that it split properly
	training_set, testing_set, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=50)
	print("Training set samples:", len(training_set), ", Testing set samples:", len(testing_set))

	# Filter out words that are too infrequent or too frequent
	vectorizer = TfidfVectorizer(min_df=10, max_df=0.90)

	Perceptron_clf = Pipeline([('vect', vectorizer),
				   ('clf', Perceptron(tol=1e-3)),
				   ])

	# Use a grid search to find optimal parameters
	Perceptron_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [1e-2, 1e-3, 1e-4],
        'clf__tol': [1e-2, 1e-3, 1e-4],
        }
	Perceptron_grid_search = GridSearchCV(Perceptron_clf, Perceptron_parameters, n_jobs=-1)
	Perceptron_grid_search.fit(training_set, y_train)

	# Print the results of the grid search
	n_candidates = len(Perceptron_grid_search.cv_results_['params'])
	for i in range(n_candidates):
		print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                % (Perceptron_grid_search.cv_results_['params'][i],
                    Perceptron_grid_search.cv_results_['mean_test_score'][i],
                    Perceptron_grid_search.cv_results_['std_test_score'][i]))

	# Perceptron_clf.fit(training_set, y_train)

	Perceptron_y_predicted = Perceptron_grid_search.predict(testing_set)

	# print Perceptron metrics report
	print(metrics.classification_report(y_test, Perceptron_y_predicted,
	                                    target_names=dataset.target_names))

	# print Perceptron accuracy
	print("Perceptron accuracy:", np.mean(Perceptron_y_predicted == y_test))

	MNB_clf = Pipeline([('vect', vectorizer),
						('clf', MultinomialNB()),
						])

	# Use a grid search to find optimal parameters
	MNB_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [1e-2, 1e-3, 1e-4],
        }
	MNB_grid_search = GridSearchCV(MNB_clf, MNB_parameters, n_jobs=-1)
	MNB_grid_search.fit(training_set, y_train)

	# Print the results of the grid search
	n_candidates = len(MNB_grid_search.cv_results_['params'])
	for i in range(n_candidates):
		print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                % (MNB_grid_search.cv_results_['params'][i],
                    MNB_grid_search.cv_results_['mean_test_score'][i],
                    MNB_grid_search.cv_results_['std_test_score'][i]))

	# MNB_clf.fit(training_set, y_train)

	MNB_y_predicted = MNB_grid_search.predict(testing_set)

	# Print MNB metrics report
	print(metrics.classification_report(y_test, MNB_y_predicted,
	                                    target_names=dataset.target_names))

	# print MNB accuracy
	print("MultinomialNB accuracy:", np.mean(MNB_y_predicted == y_test))

	LinearSVC_clf = Pipeline([
	        ('vect', vectorizer),
	        ('clf', LinearSVC(dual=False)),
	    	])

	# Use a grid search to find optimal parameters
	LinearSVC_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__penalty': ['l1', 'l2'],
        'clf__tol': [1e-2, 1e-3, 1e-4],
        'clf__C': [0.1, 1.0, 10.0, 100, 1000]
        }
	LinearSVC_grid_search = GridSearchCV(LinearSVC_clf, LinearSVC_parameters, n_jobs=-1)
	LinearSVC_grid_search.fit(training_set, y_train)

	# Print the results of the grid search
	n_candidates = len(LinearSVC_grid_search.cv_results_['params'])
	for i in range(n_candidates):
		print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                % (LinearSVC_grid_search.cv_results_['params'][i],
                    LinearSVC_grid_search.cv_results_['mean_test_score'][i],
                    LinearSVC_grid_search.cv_results_['std_test_score'][i]))

	# LinearSVC_clf.fit(training_set, y_train)

	LinearSVC_y_predicted = LinearSVC_grid_search.predict(testing_set)

	# print LinearSVC metrics report
	print(metrics.classification_report(y_test, LinearSVC_y_predicted,
	                                    target_names=dataset.target_names))

	# print LinearSVC accuracy
	print("LinearSVC accuracy:", np.mean(LinearSVC_y_predicted == y_test))
