-- Initialize Einstein database
-- This script is run when the PostgreSQL container starts for the first time

-- Create database if it doesn't exist (this is handled by POSTGRES_DB env var)
-- CREATE DATABASE einstein_db;

-- Create extensions that might be useful for scientific data
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "hstore";

-- Create user if needed (this is handled by POSTGRES_USER env var)
-- CREATE USER einstein_user WITH PASSWORD 'einstein_password';
-- GRANT ALL PRIVILEGES ON DATABASE einstein_db TO einstein_user;
