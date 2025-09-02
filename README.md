# 🏋️‍♀️ Smart Fitness Tracker

A modern web application that predicts calories burned during workouts using artificial neural networks. Built with Flask, featuring real-time predictions, beautiful data visualizations, and user session management.

![Smart Fitness Tracker](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🤖 **AI-Powered Predictions** - Advanced neural network model for accurate calorie estimation
- 📊 **Interactive Charts** - Beautiful weekly progress visualization with Chart.js
- 👤 **User Sessions** - Individual data isolation using localStorage sessions
- 📱 **Responsive Design** - Works perfectly on desktop and mobile devices
- ⚡ **Real-time Updates** - Instant chart and table updates after predictions
- 🗃️ **Data Management** - View, track, and delete fitness sessions
- 🎨 **Modern UI** - Clean, professional interface with smooth animations

## 🚀 Live Demo

[Try it live on Railway](https://your-app-name.railway.app) *(Deploy to get your URL)*

## 📸 Screenshots

### Main Dashboard
![Dashboard](https://via.placeholder.com/800x400/556B2F/FFFFFF?text=Smart+Fitness+Tracker+Dashboard)

### Weekly Progress Chart
![Chart](https://via.placeholder.com/800x300/6B8E23/FFFFFF?text=Weekly+Calories+Chart)

## 🛠️ Tech Stack

**Backend:**
- Python 3.11
- Flask 2.3.3
- SQLAlchemy (PostgreSQL/SQLite)
- NumPy for ML predictions
- UUID for session management

**Frontend:**
- HTML5 & CSS3
- Bootstrap 5.3.2
- Chart.js for visualizations
- Font Awesome icons
- Vanilla JavaScript

**Deployment:**
- Railway (recommended)
- Gunicorn WSGI server
- PostgreSQL database

## 📋 Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KasturiSnehaReddy/Smart_Fitness_Tracker.git
cd Smart_Fitness_Tracker
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
Navigate to `http://localhost:5000`

## 🎯 Usage

### Making Predictions
1. Fill in your fitness data:
   - Gender, Age, Height, Weight
   - Workout Duration
   - Heart Rate during exercise
   - Body Temperature
2. Click "Predict Calories"
3. View results in the dashboard

### Viewing Progress
- **Weekly Chart**: See calories burned across 52 weeks
- **Yearly Summary**: Total calories by year
- **Session History**: Detailed list of all workouts

### Managing Data
- **Delete Sessions**: Remove individual workout records
- **Automatic Totals**: Charts update automatically
- **Persistent Sessions**: Data saved across browser sessions

## 🧠 Machine Learning Model

The application uses a trained Artificial Neural Network (ANN) with:
- **Input Features**: Gender, Age, Height, Weight, Duration, Heart Rate, Body Temperature
- **Architecture**: Multi-layer perceptron with ReLU activation
- **Preprocessing**: StandardScaler normalization
- **Output**: Predicted calories burned

### Model Files
- `checkpoints/best_model.npz` - Trained model weights
- `checkpoints/scaler.npz` - Feature scaling parameters

## 📁 Project Structure

```
Smart_Fitness_Tracker/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Railway deployment config
├── runtime.txt           # Python version
├── README.md             # Project documentation
├── checkpoints/          # ML model files
│   ├── best_model.npz
│   └── scaler.npz
├── src/                  # Source data
│   ├── calories.csv
│   ├── model.py
│   └── training_results.png
├── templates/            # HTML templates
│   └── index.html
└── instance/            # Database files
    └── fitness.db
```

## 🚢 Deployment

### Deploy to Railway (Recommended)

1. **Fork this repository**
2. **Visit [railway.app](https://railway.app)**
3. **Connect your GitHub account**
4. **Create new project from GitHub repo**
5. **Railway automatically deploys!**

### Deploy to Heroku

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

### Environment Variables

For production deployment, Railway automatically provides:
- `DATABASE_URL` - PostgreSQL connection string
- `PORT` - Application port

## 🔒 Session Management

Each user gets a unique session ID stored in localStorage:
- **Privacy**: Users only see their own data
- **Persistence**: Data survives browser restarts
- **Isolation**: Complete separation between users
- **No Registration**: Simple UUID-based system

## 🎨 Features in Detail

### Responsive Design
- Mobile-first approach
- Bootstrap 5 grid system
- Touch-friendly interface
- Optimized for all screen sizes

### Data Visualization
- Interactive weekly charts
- Smooth animations
- Color-coded progress
- Hover tooltips

### User Experience
- Loading indicators
- Success/error notifications
- Smooth scrolling
- Keyboard navigation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Fitness tracking data for model training
- **Bootstrap**: Responsive UI framework
- **Chart.js**: Beautiful chart visualizations
- **Railway**: Excellent deployment platform
- **Font Awesome**: Amazing icon library

## 📞 Contact

**Kasturi Sneha Reddy**
- GitHub: [@KasturiSnehaReddy](https://github.com/KasturiSnehaReddy)
- Project: [Smart_Fitness_Tracker](https://github.com/KasturiSnehaReddy/Smart_Fitness_Tracker)

## 🔮 Future Enhancements

- [ ] Exercise type categorization
- [ ] Goal setting and tracking
- [ ] Data export functionality
- [ ] Social sharing features
- [ ] Nutrition tracking
- [ ] Workout recommendations
- [ ] Mobile app version

---

**⭐ Star this repository if you found it helpful!**