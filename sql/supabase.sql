-- Run once in your Supabase project's SQL Editor.
-- The Flask server connects with a database URI; browsers never query this table.
create schema if not exists extensions;
create extension if not exists vector with schema extensions;

create table if not exists public.notebook_chunks (
  notebook text not null,
  chunk_id text not null,
  source text not null,
  content text not null,
  embedding_model text not null,
  embedding extensions.vector(768) not null,
  source_fingerprint text not null,
  primary key (notebook, chunk_id)
);

create index if not exists notebook_chunks_lookup
  on public.notebook_chunks (notebook, embedding_model);

create index if not exists notebook_chunks_source
  on public.notebook_chunks (notebook, source);

-- No browser-facing policy is needed: all access goes through the Flask server.
alter table public.notebook_chunks enable row level security;
revoke all on public.notebook_chunks from anon, authenticated;
