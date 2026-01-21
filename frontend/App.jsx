import { useState, useEffect } from 'react';
import { Check, X, HelpCircle, Save, RefreshCw, Undo, ChevronLeft, ChevronRight } from 'lucide-react';

// API configuration
const API_BASE_URL = 'http://localhost:8000';

const API = {
  async loadDataset(collectionName) {
    const response = await fetch(`${API_BASE_URL}/dataset/${collectionName}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  async saveVerifications(collectionName, changes) {
    const response = await fetch(`${API_BASE_URL}/dataset/${collectionName}/batch-verify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ changes }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
};

function App() {
  const [collectionName, setCollectionName] = useState('');
  const [dataset, setDataset] = useState([]);
  const [verifications, setVerifications] = useState({});
  const [changeHistory, setChangeHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // Get collection name from URL or use hardcoded default
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const collection = params.get('collection') || 'sample_qa_dataset';
    setCollectionName(collection);
  }, []);

  // Load dataset when collection name is set
  useEffect(() => {
    if (collectionName) {
      loadDataset();
    }
  }, [collectionName]);

  const loadDataset = async () => {
    setLoading(true);
    try {
      const data = await API.loadDataset(collectionName);
      
      // Initialize dataset with sample data if API fails
      if (!data || data.length === 0) {
        // Sample hardcoded data
        const sampleData = [
          {
            _id: '1',
            question: 'What is the capital of France?',
            reasoning: 'France is a country in Western Europe, and Paris has been its capital since the 12th century.',
            answer: 'Paris',
            verified: null
          },
          {
            _id: '2',
            question: 'What is 2 + 2?',
            A: '3',
            B: '4',
            C: '5',
            D: '6',
            answer: 'B) 4',
            verified: null
          },
          {
            _id: '3',
            question: 'Which planet is known as the Red Planet?',
            reasoning: 'Mars appears red due to iron oxide on its surface.',
            answer: 'Mars',
            verified: null
          },
          {
            _id: '4',
            question: 'What is the largest ocean on Earth?',
            A: 'Atlantic Ocean',
            B: 'Indian Ocean',
            C: 'Pacific Ocean',
            D: 'Arctic Ocean',
            answer: 'C) Pacific Ocean',
            verified: null
          },
          {
            _id: '5',
            question: 'Who wrote "Romeo and Juliet"?',
            reasoning: 'William Shakespeare wrote this famous tragedy in the early 1590s.',
            answer: 'William Shakespeare',
            verified: null
          }
        ];
        setDataset(sampleData);
        showMessage('Using sample data (API not available)', 'info');
      } else {
        setDataset(data);
        showMessage(`Loaded ${data.length} items from ${collectionName}`, 'success');
      }
      
      setChangeHistory([]);
      setCurrentPage(1);
    } catch (error) {
      console.error('Failed to load dataset:', error);
      // Load sample data on error
      const sampleData = [
        {
          _id: '1',
          question: 'What is the capital of France?',
          reasoning: 'France is a country in Western Europe, and Paris has been its capital since the 12th century.',
          answer: 'Paris',
          verified: null
        },
        {
          _id: '2',
          question: 'What is 2 + 2?',
          A: '3',
          B: '4',
          C: '5',
          D: '6',
          answer: 'B) 4',
          verified: null
        }
      ];
      setDataset(sampleData);
      showMessage('Using sample data (Error loading from API)', 'info');
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (text, type = 'info') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 3000);
  };

  const recordChange = (itemId, oldValue, newValue) => {
    setChangeHistory(prev => [...prev, {
      itemId,
      old: oldValue,
      new: newValue,
      timestamp: new Date()
    }]);
  };

  const updateVerification = (itemId, status) => {
    const oldStatus = verifications[itemId] || null;
    
    if (oldStatus !== status) {
      recordChange(itemId, oldStatus, status);
      setVerifications(prev => ({
        ...prev,
        [itemId]: status
      }));
    }
  };

  const undoLastChange = () => {
    if (changeHistory.length === 0) {
      showMessage('Nothing to undo', 'info');
      return;
    }

    const lastChange = changeHistory[changeHistory.length - 1];
    setChangeHistory(prev => prev.slice(0, -1));
    
    if (lastChange.old === null) {
      setVerifications(prev => {
        const next = { ...prev };
        delete next[lastChange.itemId];
        return next;
      });
    } else {
      setVerifications(prev => ({
        ...prev,
        [lastChange.itemId]: lastChange.old
      }));
    }

    showMessage('Undone last change', 'success');
  };

  const saveVerifications = async () => {
    if (changeHistory.length === 0) {
      showMessage('No changes to save', 'info');
      return;
    }

    try {
      await API.saveVerifications(collectionName, changeHistory);
      setChangeHistory([]);
      showMessage('Verifications saved successfully!', 'success');
    } catch (error) {
      console.error('Failed to save verifications:', error);
      showMessage('Error saving: ' + error.message, 'error');
    }
  };

  const getStats = () => {
    const verified = Object.values(verifications).filter(v => v === 'approved').length;
    const rejected = Object.values(verifications).filter(v => v === 'rejected').length;
    const flagged = Object.values(verifications).filter(v => v === 'flagged').length;
    const pending = dataset.length - verified - rejected - flagged;
    
    return { verified, rejected, flagged, pending, total: dataset.length };
  };

  const getPaginatedData = () => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return dataset.slice(startIndex, endIndex);
  };

  const getMiddleFields = (item) => {
    const excludeKeys = ['_id', 'question', 'answer', 'verified'];
    return Object.keys(item).filter(key => !excludeKeys.includes(key));
  };

  const renderMiddleContent = (item) => {
    const middleFields = getMiddleFields(item);
    
    return middleFields.map(field => {
      if (['A', 'B', 'C', 'D'].includes(field)) {
        return null; // Will render MCQ separately
      }
      return (
        <div key={field} className="mb-2">
          <span className="font-medium text-gray-600 capitalize">{field}: </span>
          <span className="text-gray-800">{item[field]}</span>
        </div>
      );
    });
  };

  const renderMCQ = (item) => {
    const hasOptions = ['A', 'B', 'C', 'D'].some(opt => item[opt]);
    
    if (!hasOptions) return null;
    
    return (
      <div className="grid grid-cols-2 gap-2 mb-2">
        {['A', 'B', 'C', 'D'].map(opt => item[opt] && (
          <div key={opt} className="p-2 bg-gray-50 rounded border border-gray-200">
            <span className="font-semibold text-blue-600">{opt})</span> {item[opt]}
          </div>
        ))}
      </div>
    );
  };

  const getStatusColor = (itemId) => {
    const status = verifications[itemId];
    if (status === 'approved') return 'bg-green-50 border-l-4 border-green-500';
    if (status === 'rejected') return 'bg-red-50 border-l-4 border-red-500';
    if (status === 'flagged') return 'bg-yellow-50 border-l-4 border-yellow-500';
    return 'bg-white';
  };

  const stats = getStats();
  const paginatedData = getPaginatedData();
  const totalPages = Math.ceil(dataset.length / itemsPerPage);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 mx-auto mb-4 animate-spin text-blue-500" />
          <p className="text-gray-600">Loading dataset...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">
            ✓ Dataset Verification Tool
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Collection: <span className="font-mono font-semibold">{collectionName}</span>
          </p>
        </div>
      </div>

      {/* Message Toast */}
      {message && (
        <div className="fixed top-4 right-4 z-50">
          <div className={`px-6 py-3 rounded-lg shadow-lg ${
            message.type === 'success' ? 'bg-green-500' :
            message.type === 'error' ? 'bg-red-500' : 'bg-blue-500'
          } text-white`}>
            {message.text}
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Status Banner */}
        <div className={`mb-6 p-4 rounded-lg ${
          changeHistory.length > 0 ? 'bg-yellow-50 border border-yellow-200' : 'bg-green-50 border border-green-200'
        }`}>
          <p className={`text-sm font-medium ${
            changeHistory.length > 0 ? 'text-yellow-800' : 'text-green-800'
          }`}>
            {changeHistory.length > 0 
              ? `⚠️ ${changeHistory.length} unsaved changes`
              : `✓ No pending changes`
            }
          </p>
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <button
            onClick={saveVerifications}
            disabled={changeHistory.length === 0}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
          >
            <Save className="w-5 h-5" />
            Save Verifications
          </button>
          <button
            onClick={undoLastChange}
            disabled={changeHistory.length === 0}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
          >
            <Undo className="w-5 h-5" />
            Undo Last Change
          </button>
          <button
            onClick={loadDataset}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
          >
            <RefreshCw className="w-5 h-5" />
            Refresh Dataset
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <p className="text-sm text-gray-600">Total</p>
            <p className="text-3xl font-bold text-gray-900">{stats.total}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <p className="text-sm text-gray-600">Verified ✓</p>
            <p className="text-3xl font-bold text-green-600">{stats.verified}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <p className="text-sm text-gray-600">Rejected ✗</p>
            <p className="text-3xl font-bold text-red-600">{stats.rejected}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <p className="text-sm text-gray-600">Flagged ?</p>
            <p className="text-3xl font-bold text-yellow-600">{stats.flagged}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <p className="text-sm text-gray-600">Pending</p>
            <p className="text-3xl font-bold text-gray-600">{stats.pending}</p>
          </div>
        </div>

        {/* Dataset Items */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold">
              Verification Items (Showing {paginatedData.length} of {dataset.length})
            </h2>
          </div>

          <div className="divide-y divide-gray-200">
            {paginatedData.length === 0 ? (
              <div className="p-12 text-center text-gray-500">
                No items found in dataset
              </div>
            ) : (
              paginatedData.map((item) => (
                <div key={item._id} className={`p-6 transition ${getStatusColor(item._id)}`}>
                  <div className="grid grid-cols-12 gap-4">
                    {/* Question */}
                    <div className="col-span-12 md:col-span-3">
                      <div className="font-semibold text-gray-700 text-sm mb-1">Question</div>
                      <div className="text-gray-900">{item.question}</div>
                    </div>

                    {/* Middle Content (Reasoning/MCQ) */}
                    <div className="col-span-12 md:col-span-4">
                      {renderMCQ(item)}
                      {renderMiddleContent(item)}
                    </div>

                    {/* Answer */}
                    <div className="col-span-12 md:col-span-3">
                      <div className="font-semibold text-gray-700 text-sm mb-1">Answer</div>
                      <div className="text-gray-900 font-medium">{item.answer}</div>
                    </div>

                    {/* Buttons */}
                    <div className="col-span-12 md:col-span-2 flex gap-2 justify-end items-start">
                      <button
                        onClick={() => updateVerification(item._id, 'approved')}
                        className={`p-3 rounded-lg transition ${
                          verifications[item._id] === 'approved'
                            ? 'bg-green-600 text-white'
                            : 'bg-green-100 text-green-700 hover:bg-green-200'
                        }`}
                        title="Verify as Correct"
                      >
                        <Check className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => updateVerification(item._id, 'rejected')}
                        className={`p-3 rounded-lg transition ${
                          verifications[item._id] === 'rejected'
                            ? 'bg-red-600 text-white'
                            : 'bg-red-100 text-red-700 hover:bg-red-200'
                        }`}
                        title="Reject as Incorrect"
                      >
                        <X className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => updateVerification(item._id, 'flagged')}
                        className={`p-3 rounded-lg transition ${
                          verifications[item._id] === 'flagged'
                            ? 'bg-yellow-600 text-white'
                            : 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200'
                        }`}
                        title="Flag for Review"
                      >
                        <HelpCircle className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="p-6 border-t border-gray-200 flex justify-between items-center">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>
              
              <span className="text-gray-700">
                Page {currentPage} of {totalPages}
              </span>
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;