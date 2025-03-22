import React from 'react';
import { QueryEditorProps } from '@grafana/data';
import { Alert } from '@grafana/ui';
import { TemplarVersionDataSource } from '../datasource';
import { TemplarVersionDataSourceOptions, TemplarVersionQuery } from '../types';

type Props = QueryEditorProps<TemplarVersionDataSource, TemplarVersionQuery, TemplarVersionDataSourceOptions>;

export function QueryEditor({ datasource, onChange, onRunQuery }: Props) {
  return (
    <div className="gf-form">
      <Alert title="" severity="info">
        This datasource fetches the Templar version information from {datasource.url}. No configuration is needed for the query.
      </Alert>
    </div>
  );
}