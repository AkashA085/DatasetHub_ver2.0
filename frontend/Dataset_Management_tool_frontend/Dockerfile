# Build Stage
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies using clean install for reproducible builds
COPY package*.json ./
RUN npm ci

# Copy the rest of the source code
COPY . .

# Build the Vite application for production
RUN npm run build

# Production Stage
FROM nginx:alpine

# Copy the built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy the custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
