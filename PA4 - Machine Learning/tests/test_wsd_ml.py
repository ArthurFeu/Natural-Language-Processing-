import pytest
import numpy
import sys
sys.path.append('C:\\Users\\arthu\\Documents\\VCU Semester\\CMSC 416 - Intro to Natural Language Processing\\PA4 - Machine Learning\\src') 
from unittest.mock import mock_open, Mock, patch
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import runpy
import wsd_ml

class TestCleanContext:
	def test_clean_context_with_hello_word(self):
		input_text = "Hello, world! <html>This is a test.</html>"
		expected_output = "hello world test"
		assert wsd_ml.clean_context(input_text) == expected_output
	
	def test_clean_context_with_no_html(self):
		input_text = "Hello, world!"
		expected_output = "hello world"
		assert wsd_ml.clean_context(input_text) == expected_output
  
	def test_clean_context_with_no_html_and_no_punctuation(self):
		input_text = "Hello world"
		expected_output = "hello world"
		assert wsd_ml.clean_context(input_text) == expected_output
  
	def test_clean_context_with_multiple_html_tags(self):
		input_text = "Hello, world! <html>This is a test.</html> <html>Another test.</html>"
		expected_output = "hello world test another test"
		assert wsd_ml.clean_context(input_text) == expected_output
  
@pytest.fixture
def mock_clean_context(mocker):
    # Assign a function to side_effect to dictate how a mock behaves dynamically
    # Lambda is an anonymous function used for simple, one-expression functions
    mocker.patch('wsd_ml.clean_context', side_effect=lambda x: x + " Cleaned!")

class TestExtractContextFromFile:
    def test_extract_context_from_test(self, mocker, mock_clean_context):
        mocked_file_content = '''
        <instance id="001"><context>Hello, world! This is a test.</context></instance>
        <instance id="002"><context>Another example here.</context></instance>
        '''
        mocker.patch('builtins.open', mock_open(read_data=mocked_file_content))

        expected_output = {
            '001': 'Hello, world! This is a test. Cleaned!',
            '002': 'Another example here. Cleaned!'
        }
        
        result = wsd_ml.extract_context_from_test('dummy_path.txt')
        
        assert result == expected_output
        
class TestExtractContextsAndSenseidVector:
	def test_extract_contexts_and_senseid_vector(self, mocker, mock_clean_context):
		# Mock data simulating the file content
		mocked_file_content = '''
		<instance id="001"><answer instance="001" senseid="phone"/> <context>Hello, world! This is a test.</context></instance>
		<instance id="002"><answer instance="002" senseid="product"/> <context>Another example here.</context></instance>
		'''
		mocker.patch('builtins.open', mock_open(read_data=mocked_file_content))

		expected_context_dict = {
			'001': 'Hello, world! This is a test. Cleaned!',
			'002': 'Another example here. Cleaned!'
		}
		expected_senseid_vector = ['phone', 'product']
		
		context_dict, senseid_vector = wsd_ml.extract_contexts_and_senseid_vector('dummy_path.txt')
		
		# Two outputs are returned, so we need to check both
		assert context_dict == expected_context_dict
		assert senseid_vector == expected_senseid_vector
  
class TestPredictAndPrintAnswers:
	def test_model_predict_and_vectorizer_transform(self, mocker):
		# Create mock objects for the vectorizer and model
		mock_vectorizer = Mock()
		mock_model = Mock()

		mock_vectorizer.transform.return_value = ['vectorized data']
		mock_model.predict.return_value = ['phone']
  
		# Setup to capture the print output
		captured_output = StringIO()
		sys.stdout = captured_output
		
		test_contexts = {'001': 'some context data'}
		
		wsd_ml.predict_and_print_answers(test_contexts, mock_model, mock_vectorizer)
		
		# Check outputs
		assert "<answer instance=\"001\" senseid=\"phone\"/>" in captured_output.getvalue()

class TestMain:
	def test_main_without_model_specified(self):
		# Mock sys.argv to simulate command line input
		with patch.object(sys, 'argv', ['wsd_ml.py', 'line-train.txt', 'line-test.txt']):
			# Mock the extract functions and model/Vectorizer behavior
			with patch('wsd_ml.extract_contexts_and_senseid_vector') as mock_extract_contexts, \
				patch('wsd_ml.extract_context_from_test') as mock_extract_test, \
				patch('wsd_ml.CountVectorizer') as mock_vectorizer_class, \
				patch('wsd_ml.MultinomialNB', Mock(return_value=Mock(spec=MultinomialNB))) as mock_nb_class, \
				patch('wsd_ml.predict_and_print_answers') as mock_predict_print:
				
				# Setup mock returns
				mock_extract_contexts.return_value = ({"001": "hello word"}, ["phone"])
				mock_extract_test.return_value = {"002": "yes, we test"}
				mock_vectorizer_instance = Mock(spec=CountVectorizer)
				mock_vectorizer_class.return_value = mock_vectorizer_instance
				mock_vectorizer_instance.fit_transform.return_value = 'vectorized_data'
				mock_model_instance = Mock()
				mock_nb_class.return_value = mock_model_instance
				
				wsd_ml.main()

				# Check that the right classes were used
				mock_vectorizer_class.assert_called_once()
				mock_nb_class.assert_called_once()  # Since 'nb' is the default choice

				# Verify interactions
				mock_model_instance.fit.assert_called_once_with('vectorized_data', ["phone"])
				mock_predict_print.assert_called_once_with({"002": "yes, we test"}, mock_model_instance, mock_vectorizer_instance)
    
	def test_main_with_logreg_model(self):
		# Mock sys.argv to simulate command line input
		with patch.object(sys, 'argv', ['wsd_ml.py', 'line-train.txt', 'line-test.txt', 'logreg']):
			# Mock the extract functions and model/Vectorizer behavior
			with patch('wsd_ml.extract_contexts_and_senseid_vector') as mock_extract_contexts, \
				patch('wsd_ml.extract_context_from_test') as mock_extract_test, \
				patch('wsd_ml.CountVectorizer') as mock_vectorizer_class, \
				patch('wsd_ml.LogisticRegression', Mock(return_value=Mock(spec=LogisticRegression))) as mock_logreg_class, \
				patch('wsd_ml.predict_and_print_answers') as mock_predict_print:

				# Setup mock returns
				mock_extract_contexts.return_value = ({"001": "hello word"}, ["phone"])
				mock_extract_test.return_value = {"002": "yes, we test"}
				mock_vectorizer_instance = Mock(spec=CountVectorizer)
				mock_vectorizer_class.return_value = mock_vectorizer_instance
				mock_vectorizer_instance.fit_transform.return_value = 'vectorized_data'
				mock_model_instance = Mock()
				mock_logreg_class.return_value = mock_model_instance

				wsd_ml.main()

				# Check that the right classes were used
				mock_vectorizer_class.assert_called_once()
				mock_logreg_class.assert_called_once()  # Since 'logreg' was specified
    
	def test_main_with_svc_model(self):
		# Mock sys.argv to simulate command line input
		with patch.object(sys, 'argv', ['wsd_ml.py', 'line-train.txt', 'line-test.txt', 'svm']):
			# Mock the extract functions and model/Vectorizer behavior
			with patch('wsd_ml.extract_contexts_and_senseid_vector') as mock_extract_contexts, \
					patch('wsd_ml.extract_context_from_test') as mock_extract_test, \
					patch('wsd_ml.CountVectorizer') as mock_vectorizer_class, \
					patch('wsd_ml.SVC', Mock(return_value=Mock(spec=SVC))) as mock_svc_class, \
					patch('wsd_ml.predict_and_print_answers') as mock_predict_print:

				# Setup mock returns
				mock_extract_contexts.return_value = ({"001": "hello word", "002": "yes, we test"}, ["phone", "product"])
				mock_extract_test.return_value = {"003": "test context"}
				mock_vectorizer_instance = Mock(spec=CountVectorizer)
				mock_vectorizer_class.return_value = mock_vectorizer_instance
				mock_vectorizer_instance.fit_transform.return_value = numpy.array([[1, 2], [3, 4]])  

				# Mock the model instance
				mock_model_instance = Mock()
				mock_svc_class.return_value = mock_model_instance

				wsd_ml.main()

				actual_args, _ = mock_model_instance.fit.call_args
				expected_args = (numpy.array([[1, 2], [3, 4]]), ["phone", "product"])

				assert numpy.array_equal(actual_args[0], expected_args[0])
				assert actual_args[1] == expected_args[1]
				mock_predict_print.assert_called_once_with({"003": "test context"}, mock_model_instance, mock_vectorizer_instance)



    
	def test_main_with_invalid_usage(self):
		with patch.object(sys, 'argv', ['wsd_ml.py']):
			with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
				with pytest.raises(SystemExit) as exception:
					wsd_ml.main()
     
				assert exception.type == SystemExit
				assert exception.value.code == 1
				assert "Usage: python3 wsd_ml.py line-train.txt line-test.txt [OPTIONAL: ml-model] > my-line-answers.txt" in mock_stdout.getvalue()


@pytest.fixture(autouse=True)
def restore_stdout():
    # Restore stdout after each test case
    yield
    sys.stdout = sys.__stdout__