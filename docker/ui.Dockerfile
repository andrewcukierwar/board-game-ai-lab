# docker/ui.Dockerfile  ── Vite + React

# ---------- build stage ----------
FROM node:20 AS build
WORKDIR /app
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build        # outputs to /app/dist

# ---------- serve stage ----------
FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
