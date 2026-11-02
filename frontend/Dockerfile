FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Run unprivileged; vite still compiles vite.config.ts into .vite-temp on every start, so that dir must be writable
RUN mkdir -p node_modules/.vite-temp && chown node node_modules/.vite-temp
USER node

# vite preview serves the build and proxies /api to BACKEND_URL (see vite.config.ts)
EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
