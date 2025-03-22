import { DataSourcePlugin } from '@grafana/data';
import { TemplarVersionDataSource } from './datasource';
import { ConfigEditor } from './components/ConfigEditor';
import { QueryEditor } from './components/QueryEditor';
import { TemplarVersionQuery, TemplarVersionDataSourceOptions } from './types';

export const plugin = new DataSourcePlugin<TemplarVersionDataSource, TemplarVersionQuery, TemplarVersionDataSourceOptions>(
  TemplarVersionDataSource
)
  .setConfigEditor(ConfigEditor)
  .setQueryEditor(QueryEditor);