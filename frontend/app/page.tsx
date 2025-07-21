'use client';

import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Leaf, 
  Download, 
  Settings, 
  Search,
  User,
  CheckCircle,
  AlertTriangle,
  Info,
  Clock,
  Activity,
  Image,
  BarChart3,
  Eye,
  FileText
} from 'lucide-react';

export default function AgridroneDashboard() {
  const [activeTab, setActiveTab] = useState('ndvi');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [ndviImageUrl, setNdviImageUrl] = useState<string | null>(null);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResults(null);
      setNdviImageUrl(null);
      setAnnotatedImageUrl(null);
    }
  };

  const handleNdviPrediction = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch('http://localhost:8001/predict_ndvi/', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setNdviImageUrl(url);
        
        // Extract filename from response headers if available
        const contentDisposition = response.headers.get('content-disposition');
        const filename = contentDisposition 
          ? contentDisposition.split('filename=')[1]?.replace(/"/g, '') 
          : 'ndvi_results.zip';
        
        setResults({
          type: 'ndvi',
          filename: filename,
          message: 'NDVI prediction completed successfully'
        });
      } else {
        throw new Error('NDVI prediction failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('NDVI prediction failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleYoloDetection = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch('http://localhost:8001/predict_yolo/', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setResults({
          type: 'yolo',
          data: data,
          message: 'YOLO detection completed successfully'
        });
      } else {
        throw new Error('YOLO detection failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('YOLO detection failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleYoloImageDetection = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch('http://localhost:8001/predict_yolo_image/', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setAnnotatedImageUrl(url);
        
        setResults({
          type: 'yolo_image',
          message: 'YOLO detection with image completed successfully'
        });
      } else {
        throw new Error('YOLO detection failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('YOLO detection failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePipelinePrediction = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch('http://localhost:8001/predict_pipeline/', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setResults({
          type: 'pipeline',
          data: data,
          message: 'Full pipeline completed successfully'
        });
      } else {
        throw new Error('Pipeline prediction failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Pipeline prediction failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePipelineImagePrediction = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch('http://localhost:8001/predict_pipeline_image/', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setAnnotatedImageUrl(url);
        
        setResults({
          type: 'pipeline_image',
          message: 'Full pipeline with image completed successfully'
        });
      } else {
        throw new Error('Pipeline prediction failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Pipeline prediction failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadNdviResults = () => {
    if (ndviImageUrl) {
      const a = document.createElement('a');
      a.href = ndviImageUrl;
      a.download = 'ndvi_results.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const downloadAnnotatedImage = () => {
    if (annotatedImageUrl) {
      const a = document.createElement('a');
      a.href = annotatedImageUrl;
      a.download = 'annotated_image.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center">
                <Leaf className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">Agridrone Analysis</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-green-600" />
                Analysis Options
              </h3>
              
              <div className="space-y-4">
                <div className="p-3 bg-green-50 rounded-lg">
                  <div className="text-sm text-green-600 font-medium mb-1">NDVI Prediction</div>
                  <div className="text-xs text-green-700">RGB → NDVI visualization + band data</div>
                </div>
                
                <div className="p-3 bg-blue-50 rounded-lg">
                  <div className="text-sm text-blue-600 font-medium mb-1">YOLO Detection</div>
                  <div className="text-xs text-blue-700">4-channel TIFF → Object detection</div>
                </div>
                
                <div className="p-3 bg-purple-50 rounded-lg">
                  <div className="text-sm text-purple-600 font-medium mb-1">Full Pipeline</div>
                  <div className="text-xs text-purple-700">RGB → NDVI → YOLO detection</div>
                </div>
              </div>

              <div className="mt-6 space-y-3">
                <div className="text-xs text-gray-600">
                  <div className="font-medium mb-1">File Requirements:</div>
                  <div>• NDVI: RGB images</div>
                  <div>• YOLO: 4-channel TIFF</div>
                  <div>• Pipeline: RGB images</div>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Tab Navigation */}
            <div className="flex space-x-1 bg-white rounded-lg p-1 shadow-sm mb-6">
              {[
                { id: 'ndvi', label: 'NDVI Prediction', icon: Leaf },
                { id: 'yolo', label: 'YOLO Detection', icon: Eye },
                { id: 'pipeline', label: 'Full Pipeline', icon: BarChart3 }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-green-600 text-white'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>

            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="card"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
                {activeTab === 'ndvi' ? 'NDVI Prediction' : 
                 activeTab === 'yolo' ? 'YOLO Detection' : 'Full Pipeline'}
              </h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-green-400 transition-colors">
                    {previewUrl ? (
                      <div className="space-y-4">
                        <img src={previewUrl} alt="Preview" className="max-w-full h-64 object-cover rounded-lg mx-auto" />
                        <button
                          onClick={() => {
                            setSelectedFile(null);
                            setPreviewUrl(null);
                            setResults(null);
                            setNdviImageUrl(null);
                            setAnnotatedImageUrl(null);
                          }}
                          className="btn-secondary"
                        >
                          Remove File
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                        <div>
                          <p className="text-lg font-medium text-gray-900">
                            Upload {activeTab === 'ndvi' ? 'RGB image' : 
                                    activeTab === 'yolo' ? '4-channel TIFF' : 'RGB image'}
                          </p>
                          <p className="text-gray-500">
                            {activeTab === 'ndvi' ? 'RGB image for NDVI prediction' : 
                             activeTab === 'yolo' ? '4-channel TIFF for object detection' : 
                             'RGB image for full pipeline analysis'}
                          </p>
                        </div>
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="btn-primary"
                        >
                          Choose File
                        </button>
                      </div>
                    )}
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </div>

                  {selectedFile && (
                    <div className="mt-4 space-y-3">
                      {activeTab === 'ndvi' && (
                        <button
                          onClick={handleNdviPrediction}
                          disabled={isProcessing}
                          className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isProcessing ? 'Processing...' : 'Predict NDVI'}
                        </button>
                      )}
                      
                      {activeTab === 'yolo' && (
                        <div className="space-y-2">
                          <button
                            onClick={handleYoloDetection}
                            disabled={isProcessing}
                            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isProcessing ? 'Processing...' : 'Detect Objects (JSON)'}
                          </button>
                          <button
                            onClick={handleYoloImageDetection}
                            disabled={isProcessing}
                            className="w-full btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isProcessing ? 'Processing...' : 'Detect Objects (Image)'}
                          </button>
                        </div>
                      )}
                      
                      {activeTab === 'pipeline' && (
                        <div className="space-y-2">
                          <button
                            onClick={handlePipelinePrediction}
                            disabled={isProcessing}
                            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isProcessing ? 'Processing...' : 'Run Pipeline (JSON)'}
                          </button>
                          <button
                            onClick={handlePipelineImagePrediction}
                            disabled={isProcessing}
                            className="w-full btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isProcessing ? 'Processing...' : 'Run Pipeline (Image)'}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Results</h3>
                  
                  {isProcessing ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mx-auto mb-4"></div>
                      <p className="text-gray-600">Processing...</p>
                    </div>
                  ) : results ? (
                    <div className="space-y-4">
                      <div className="bg-green-50 p-4 rounded-lg">
                        <div className="text-sm text-green-600 font-medium">Status</div>
                        <div className="text-sm text-green-900">{results.message}</div>
                      </div>
                      
                      {results.type === 'ndvi' && ndviImageUrl && (
                        <div>
                          <div className="text-sm text-gray-600 font-medium mb-2">NDVI Results (ZIP)</div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-2">
                                <FileText className="w-4 h-4 text-gray-600" />
                                <span className="text-sm text-gray-900">NDVI visualization + band data</span>
                              </div>
                              <button
                                onClick={downloadNdviResults}
                                className="btn-secondary text-xs"
                              >
                                Download
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {results.type === 'yolo' && results.data && (
                        <div>
                          <div className="text-sm text-gray-600 font-medium mb-2">Detection Results</div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-900">
                              <div>Detections: {results.data.detections?.length || 0}</div>
                              <div>Processing Time: {results.data.processing_time?.toFixed(2) || 'N/A'}s</div>
                            </div>
                            {results.data.detections && results.data.detections.length > 0 && (
                              <div className="mt-2">
                                <div className="text-xs text-gray-600 font-medium mb-1">Detected Objects:</div>
                                <div className="space-y-1">
                                  {results.data.detections.slice(0, 5).map((detection: any, index: number) => (
                                    <div key={index} className="text-xs text-gray-700">
                                      {detection.class}: {(detection.confidence * 100).toFixed(1)}%
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {results.type === 'pipeline' && results.data && (
                        <div>
                          <div className="text-sm text-gray-600 font-medium mb-2">Pipeline Results</div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-900">
                              <div>Detections: {results.data.detections?.length || 0}</div>
                              <div>Processing Time: {results.data.processing_time?.toFixed(2) || 'N/A'}s</div>
                            </div>
                            {results.data.detections && results.data.detections.length > 0 && (
                              <div className="mt-2">
                                <div className="text-xs text-gray-600 font-medium mb-1">Detected Objects:</div>
                                <div className="space-y-1">
                                  {results.data.detections.slice(0, 5).map((detection: any, index: number) => (
                                    <div key={index} className="text-xs text-gray-700">
                                      {detection.class}: {(detection.confidence * 100).toFixed(1)}%
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {annotatedImageUrl && (
                        <div>
                          <div className="text-sm text-gray-600 font-medium mb-2">Annotated Image</div>
                          <img src={annotatedImageUrl} alt="Annotated" className="w-full rounded-lg shadow-md" />
                          <button
                            onClick={downloadAnnotatedImage}
                            className="w-full mt-2 btn-secondary text-sm"
                          >
                            Download Annotated Image
                          </button>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Leaf className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                      <p>Upload a file to see analysis results</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
} 