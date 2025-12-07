import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, Package, AlertCircle, Activity, Upload, Instagram, MessageSquare, Brain, ChevronRight, Calendar, DollarSign } from 'lucide-react';

const DemandForecastingPlatform = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Sample data - replace with API calls
  const forecastData = [
    { date: '2024-12', actual: 1200, predicted: 1180, lower: 1100, upper: 1260 },
    { date: '2025-01', actual: 1350, predicted: 1340, lower: 1250, upper: 1430 },
    { date: '2025-02', actual: 1100, predicted: 1120, lower: 1050, upper: 1190 },
    { date: '2025-03', actual: null, predicted: 1280, lower: 1200, upper: 1360 },
    { date: '2025-04', actual: null, predicted: 1450, lower: 1350, upper: 1550 },
    { date: '2025-05', actual: null, predicted: 1380, lower: 1280, upper: 1480 },
  ];

  const sentimentData = [
    { platform: 'Instagram', positive: 68, neutral: 22, negative: 10 },
    { platform: 'Reddit', positive: 52, neutral: 30, negative: 18 },
    { platform: 'Combined', positive: 60, neutral: 26, negative: 14 },
  ];

  const trendingProducts = [
    { name: 'Wireless Earbuds', score: 92, change: '+15%', sentiment: 'positive' },
    { name: 'Smart Watch', score: 88, change: '+8%', sentiment: 'positive' },
    { name: 'Portable Charger', score: 75, change: '-3%', sentiment: 'neutral' },
    { name: 'Phone Case', score: 70, change: '+5%', sentiment: 'positive' },
  ];

  const inventoryAlerts = [
    { product: 'Wireless Earbuds', status: 'low', current: 45, recommended: 150, action: 'Restock' },
    { product: 'Smart Watch', status: 'optimal', current: 120, recommended: 115, action: 'Monitor' },
    { product: 'Laptop Stand', status: 'overstock', current: 200, recommended: 80, action: 'Reduce' },
  ];

  const StatCard = ({ title, value, change, icon: Icon, color }) => (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 font-medium">{title}</p>
          <h3 className="text-3xl font-bold text-gray-900 mt-2">{value}</h3>
          <p className={`text-sm mt-2 flex items-center ${change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
            {change} <span className="text-gray-500 ml-1">vs last month</span>
          </p>
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  );

  const renderDashboard = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Forecast Accuracy" value="94.2%" change="+2.3%" icon={Brain} color="bg-gradient-to-br from-purple-500 to-purple-600" />
        <StatCard title="Predicted Sales" value="$1.28M" change="+12.5%" icon={TrendingUp} color="bg-gradient-to-br from-blue-500 to-blue-600" />
        <StatCard title="Stock Efficiency" value="87%" change="+5.1%" icon={Package} color="bg-gradient-to-br from-green-500 to-green-600" />
        <StatCard title="Active Alerts" value="3" change="-1" icon={AlertCircle} color="bg-gradient-to-br from-orange-500 to-orange-600" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Demand Forecast (Next 6 Months)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={forecastData}>
              <defs>
                <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }} />
              <Legend />
              <Area type="monotone" dataKey="upper" stroke="none" fill="#93c5fd" fillOpacity={0.2} />
              <Area type="monotone" dataKey="lower" stroke="none" fill="#93c5fd" fillOpacity={0.2} />
              <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={3} dot={{ fill: '#10b981', r: 5 }} name="Actual Sales" />
              <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={3} strokeDasharray="5 5" dot={{ fill: '#3b82f6', r: 5 }} name="Predicted Sales" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Trending Products</h3>
          <div className="space-y-4">
            {trendingProducts.map((product, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex-1">
                  <p className="font-semibold text-gray-900 text-sm">{product.name}</p>
                  <div className="flex items-center mt-1">
                    <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                      <div className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full" style={{ width: `${product.score}%` }}></div>
                    </div>
                    <span className="text-xs font-bold text-gray-600">{product.score}</span>
                  </div>
                </div>
                <span className="text-sm font-bold text-green-600 ml-3">{product.change}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Inventory Alerts</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Product</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Status</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Current Stock</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Recommended</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Action</th>
              </tr>
            </thead>
            <tbody>
              {inventoryAlerts.map((alert, idx) => (
                <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 text-sm font-medium text-gray-900">{alert.product}</td>
                  <td className="py-3 px-4">
                    <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full ${
                      alert.status === 'low' ? 'bg-red-100 text-red-700' :
                      alert.status === 'optimal' ? 'bg-green-100 text-green-700' :
                      'bg-orange-100 text-orange-700'
                    }`}>
                      {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-600">{alert.current} units</td>
                  <td className="py-3 px-4 text-sm text-gray-600">{alert.recommended} units</td>
                  <td className="py-3 px-4">
                    <button className="text-sm font-semibold text-blue-600 hover:text-blue-700 flex items-center">
                      {alert.action} <ChevronRight className="w-4 h-4 ml-1" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderSocialTrends = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-pink-500 to-purple-600 rounded-xl shadow-lg p-6 text-white">
          <Instagram className="w-8 h-8 mb-3" />
          <h3 className="text-2xl font-bold">12,450</h3>
          <p className="text-pink-100 mt-1">Instagram Mentions</p>
          <p className="text-sm text-pink-200 mt-2">+18% this week</p>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-red-600 rounded-xl shadow-lg p-6 text-white">
          <MessageSquare className="w-8 h-8 mb-3" />
          <h3 className="text-2xl font-bold">8,320</h3>
          <p className="text-orange-100 mt-1">Reddit Discussions</p>
          <p className="text-sm text-orange-200 mt-2">+12% this week</p>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl shadow-lg p-6 text-white">
          <Activity className="w-8 h-8 mb-3" />
          <h3 className="text-2xl font-bold">68%</h3>
          <p className="text-blue-100 mt-1">Positive Sentiment</p>
          <p className="text-sm text-blue-200 mt-2">+5% this week</p>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Sentiment Analysis by Platform</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sentimentData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="platform" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }} />
            <Legend />
            <Bar dataKey="positive" stackId="a" fill="#10b981" radius={[0, 0, 0, 0]} />
            <Bar dataKey="neutral" stackId="a" fill="#f59e0b" radius={[0, 0, 0, 0]} />
            <Bar dataKey="negative" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Top Instagram Hashtags</h3>
          <div className="space-y-3">
            {['#techgadgets', '#smartwatches', '#wirelessearbuds', '#techdeals', '#gadgetlover'].map((tag, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
                <span className="font-semibold text-purple-700">{tag}</span>
                <span className="text-sm text-gray-600">{(5200 - idx * 800).toLocaleString()} posts</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Reddit Hot Topics</h3>
          <div className="space-y-3">
            {['Best wireless earbuds 2025', 'Smart watch comparison', 'Budget tech recommendations', 'New gadget releases', 'Tech deals discussion'].map((topic, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gradient-to-r from-orange-50 to-red-50 rounded-lg">
                <span className="font-medium text-gray-800 text-sm">{topic}</span>
                <span className="text-xs font-semibold text-orange-600">↑ {(850 - idx * 150)} votes</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderDataUpload = () => (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Upload Historical Data</h3>
        <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer">
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-lg font-semibold text-gray-700 mb-2">Drop your CSV file here</p>
          <p className="text-sm text-gray-500 mb-4">or click to browse</p>
          <button className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors">
            Select File
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-4">Supported format: CSV with columns (date, product, sales, price)</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Data Sources Configuration</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center">
              <Instagram className="w-6 h-6 text-pink-500 mr-3" />
              <div>
                <p className="font-semibold text-gray-900">Instagram API</p>
                <p className="text-sm text-gray-500">Using Instaloader</p>
              </div>
            </div>
            <span className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm font-semibold">Connected</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center">
              <MessageSquare className="w-6 h-6 text-orange-500 mr-3" />
              <div>
                <p className="font-semibold text-gray-900">Reddit API</p>
                <p className="text-sm text-gray-500">Using PRAW</p>
              </div>
            </div>
            <span className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm font-semibold">Connected</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center">
              <Calendar className="w-6 h-6 text-blue-500 mr-3" />
              <div>
                <p className="font-semibold text-gray-900">Kaggle Dataset</p>
                <p className="text-sm text-gray-500">Sales historical data</p>
              </div>
            </div>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700">Configure</button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-xl">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">DemandIQ</h1>
                <p className="text-xs text-gray-500">Intelligent Forecasting Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="bg-green-100 text-green-700 px-4 py-2 rounded-lg text-sm font-semibold">
                Model Accuracy: 94.2%
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-1">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'forecast', label: 'Forecasting', icon: TrendingUp },
              { id: 'social', label: 'Social Trends', icon: MessageSquare },
              { id: 'inventory', label: 'Inventory', icon: Package },
              { id: 'upload', label: 'Data Upload', icon: Upload },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-6 py-4 font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'dashboard' && renderDashboard()}
        {activeTab === 'social' && renderSocialTrends()}
        {activeTab === 'upload' && renderDataUpload()}
        {(activeTab === 'forecast' || activeTab === 'inventory') && (
          <div className="text-center py-16">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
              <Brain className="w-8 h-8 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Section Under Development</h3>
            <p className="text-gray-600">This module will be available once the backend is integrated.</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default DemandForecastingPlatform;