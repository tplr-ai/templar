import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import metricsRoutes from './routes/metrics.js';
import healthRoutes from './routes/health.js';
import priceRoutes from './routes/price.js';

const app = express();

// Enable CORS
app.use(cors());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Routes
app.use('/api/metrics', metricsRoutes);
app.use('/api', healthRoutes);
app.use('/api/price', priceRoutes);

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

const PORT = process.env.PORT || 3050;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});