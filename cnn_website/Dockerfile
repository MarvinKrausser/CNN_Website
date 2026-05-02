# Step 1: Build the app
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build the project
RUN npm run build


# Step 2: Run preview server
FROM node:20-alpine

WORKDIR /app

# Copy built files and node_modules (for vite preview)
COPY --from=builder /app ./

# Expose Vite preview port
EXPOSE 4173

# Run preview server
CMD ["npm", "run", "preview", "--", "--host"]