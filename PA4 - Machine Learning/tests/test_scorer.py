import pytest
import sys
sys.path.append('C:\\Users\\arthu\\Documents\\VCU Semester\\CMSC 416 - Intro to Natural Language Processing\\PA4 - Machine Learning\\src') 
from unittest.mock import mock_open, patch, MagicMock
from hypothesis import given, strategies as st
from io import StringIO
import scorer

class TestExtractData:
	@given(file_content=st.text())
	def test_extract_data_fuzzing(self, file_content):
		# Mocking open to use Hypothesis-generated content
		with patch("builtins.open", mock_open(read_data=file_content)), \
			patch("os.path.exists", return_value=True):
			
			data = scorer.extract_data("dummy_path.txt")
			assert isinstance(data, dict)  # Simple check to ensure output is a dictionary
			for key, value in data.items():
				assert isinstance(key, str)  # Check that keys are strings (instance IDs)
				assert isinstance(value, str)  # Check that values are strings (sense IDs)
    
	def test_extract_data_with_sense_id_and_instance(self, file_content="instance and senseid"):
		with patch("builtins.open", mock_open(read_data=file_content)), \
			patch("os.path.exists", return_value=True):
			
			data = scorer.extract_data("dummy_path.txt")
			assert isinstance(data, dict)  # Simple check to ensure output is a dictionary
			for key, value in data.items():
				assert isinstance(key, str)  # Check that keys are strings (instance IDs)
				assert isinstance(value, str)  # Check that values are strings (sense IDs)
    
	def test_extract_data_with_sense_id_and_instance_and_complete_args(self, file_content='<answer instance="line-n.w8_059:8174:" senseid="phone"/>'):
		with patch("builtins.open", mock_open(read_data=file_content)), \
			patch("os.path.exists", return_value=True):
			
			data = scorer.extract_data("dummy_path.txt")
			assert isinstance(data, dict)  # Simple check to ensure output is a dictionary
			for key, value in data.items():
				assert isinstance(key, str)  # Check that keys are strings (instance IDs)
				assert isinstance(value, str)  # Check that values are strings (sense IDs)
    
	@given(file_content=st.text())
	def test_extract_data_handling_exceptions(self, file_content):
		mock_file = mock_open(read_data=file_content)
		mock_file.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
									UnicodeDecodeError("utf-16", b"", 0, 1, "invalid start byte"),
									UnicodeDecodeError("utf-8-sig", b"", 0, 1, "invalid start byte")]

		with patch("builtins.open", mock_file), \
				patch("os.path.exists", return_value=True), \
				patch('sys.stdout', new_callable=StringIO) as fake_out:
			data = scorer.extract_data("dummy_path.txt")
			assert data == {}, "Function should return an empty dictionary when exceptions occur"
			assert "Could not read file" in fake_out.getvalue(), "Function did not print the expected error message"
   
class TestCompare:
	def test_compare_answers_correct(self):
		my_answers = {'001': 'phone', '002': 'phone', '003': 'phone'}
		key_answers = {'001': 'phone', '002': 'phone', '003': 'phone'}
		correct, incorrect = scorer.compare_answers(my_answers, key_answers)
		assert correct == 3
		assert incorrect == 0
  
	def test_compare_answers_incorrect(self):
		my_answers = {'001': 'phone', '002': 'phone', '003': 'phone'}
		key_answers = {'001': 'phone', '002': 'product', '003': 'phone'}
		correct, incorrect = scorer.compare_answers(my_answers, key_answers)
		assert correct == 2
		assert incorrect == 1
  
	def test_compare_answers_key_answers_missing(self):
		my_answers = {'001': 'phone', '002': 'phone', '003': 'phone'}
		key_answers = {'001': 'phone', '003': 'phone'}
		correct, incorrect = scorer.compare_answers(my_answers, key_answers)
		assert correct == 2
		assert incorrect == 0
  
class TestCreateConfusionMatrix:
	def test_create_confusion_matrix_key_answers_missing(self):
		my_answers = {'001': 'phone', '002': 'phone', '003': 'product'}
		key_answers = {'001': 'phone', '003': 'product'}
		confusion_matrix = scorer.create_confusion_matrix(my_answers, key_answers)
		assert confusion_matrix == {'phone': {'phone': 1, 'product': 0}, 'product': {'phone': 0, 'product': 1}}
  
	def test_create_confusion_matrix_my_answers_missing(self):
		my_answers = {'001': 'phone', '003': 'product'}
		key_answers = {'001': 'phone', '002': 'product', '003': 'product'}
		confusion_matrix = scorer.create_confusion_matrix(my_answers, key_answers)
		assert confusion_matrix == {'phone': {'phone': 1, 'product': 0}, 'product': {'phone': 0, 'product': 1}}
  
	def test_create_confusion_matrix_both_answers_missing(self):
		my_answers = {'001': 'phone', '003': 'product'}
		key_answers = {'001': 'phone', '003': 'product'}
		confusion_matrix = scorer.create_confusion_matrix(my_answers, key_answers)
		assert confusion_matrix == {'phone': {'phone': 1, 'product': 0}, 'product': {'phone': 0, 'product': 1}}
  
	def test_create_confusion_matrix_empty_answers(self):
		my_answers = {}
		key_answers = {}
		confusion_matrix = scorer.create_confusion_matrix(my_answers, key_answers)
		assert confusion_matrix == {}
  
class TestPrintConfusionMatrix:
	mock_confusion_matrix = { 'phone': {'phone': 1, 'product': 0}, 'product': {'phone': 0, 'product': 1} }
	senses = sorted(mock_confusion_matrix.keys())
	
	# Setup to capture the print output
	captured_output = StringIO()
	sys.stdout = captured_output
 
	scorer.print_confusion_matrix(mock_confusion_matrix)
 
	# Check outputs
	assert "Confusion Matrix:" in captured_output.getvalue()
	assert "Actual \\ Predicted" in captured_output.getvalue()
	assert "phone" in captured_output.getvalue()
	assert "product" in captured_output.getvalue()
	assert "phone" in captured_output.getvalue()
	assert "product" in captured_output.getvalue()
	assert "1" in captured_output.getvalue()
	assert "0" in captured_output.getvalue()

class TestMain:
	def test_main_usage_message(self, capsys):
		with patch.object(sys, 'argv', ['scorer.py']):
			with pytest.raises(SystemExit) as e:
				scorer.main()
			assert e.value.code == 1
			captured = capsys.readouterr()
			assert "Usage: python3 scorer.py my-line-answers.txt line-key.txt" in captured.out

	def test_main_no_answers_extracted(self, capsys):
		mock_data = ''  # Empty data simulates no answers found
		with patch.object(sys, 'argv', ['scorer.py', 'my-line-answers.txt', 'line-key.txt']), \
			patch('builtins.open', mock_open(read_data=mock_data)), \
			patch('os.path.exists', return_value=True):
			with pytest.raises(SystemExit) as e:
				scorer.main()
			assert e.value.code == 2
			captured = capsys.readouterr()
			assert "No answers extracted from my-line-answers.txt. Please check the file format." in captured.out

	def test_main_with_answers(self, capsys):
		# Mock data simulating correct answers for "phone" and "product"
		mock_answers_data = '<answer instance="line-n.w8_059:8174:" senseid="phone"/>\n' \
							'<answer instance="line-n.w8_059:8175:" senseid="product"/>\n'
		mock_key_data = '<answer instance="line-n.w8_059:8174:" senseid="phone"/>\n' \
						'<answer instance="line-n.w8_059:8175:" senseid="product"/>\n'
		
		# Create two different mock files, one for each call of open
		mocks = [mock_open(read_data=mock_answers_data).return_value,
				mock_open(read_data=mock_key_data).return_value]

		with patch.object(sys, 'argv', ['scorer.py', 'my-line-answers.txt', 'line-key.txt']), \
			patch('builtins.open', mock_open()) as mock_file, \
			patch('os.path.exists', return_value=True):
			# Configure the side_effect to return our mocks
			mock_file.side_effect = mocks
			scorer.main()
			captured = capsys.readouterr()
			assert "phone" in captured.out
			assert "product" in captured.out
			assert "Confusion Matrix:" in captured.out
			assert "Actual \\ Predicted" in captured.out
			assert "phone" in captured.out
			assert "product" in captured.out
			assert "1" in captured.out
			assert "1" in captured.out
			assert "0" in captured.out
			assert "0" in captured.out	
   
	def test_main_with_no_valid_comparisons(self, capsys):
		# Mock data simulating no valid comparisons (different instance IDs or senseids)
		mock_answers_data = '<answer instance="line-n.w8_059:8174:" senseid="phone"/>\n'
		mock_key_data = '<answer instance="line-n.w8_059:8175:" senseid="product"/>\n'
		
		# Create two different mock files, one for each call of open
		mocks = [mock_open(read_data=mock_answers_data).return_value,
				mock_open(read_data=mock_key_data).return_value]

		with patch.object(sys, 'argv', ['scorer.py', 'my-line-answers.txt', 'line-key.txt']), \
			patch('builtins.open', mock_open()) as mock_file, \
			patch('os.path.exists', return_value=True):
			# Configure the side_effect to return our mocks
			mock_file.side_effect = mocks
			scorer.main()
			captured = capsys.readouterr()
			
			# Check that the message for no valid comparisons is in the captured output
			assert "No valid comparisons made. Please check the formats of both answer files." in captured.out

 