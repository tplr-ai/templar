import { DataSourceJsonData } from '@grafana/data';
import { DataQuery } from '@grafana/schema';

export interface TemplarVersionQuery extends DataQuery {
  endpoint?: string;
}

/**
 * These are options configured for each DataSource instance
 */
export interface TemplarVersionDataSourceOptions extends DataSourceJsonData {
  url?: string;
}