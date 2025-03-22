import React, { ChangeEvent } from 'react';
import { InlineField, Input } from '@grafana/ui';
import { DataSourcePluginOptionsEditorProps } from '@grafana/data';
import { TemplarVersionDataSourceOptions } from '../types';

interface Props extends DataSourcePluginOptionsEditorProps<TemplarVersionDataSourceOptions> {}

export function ConfigEditor(props: Props) {
  const { onOptionsChange, options } = props;

  const onUrlChange = (event: ChangeEvent<HTMLInputElement>) => {
    const jsonData = {
      ...options.jsonData,
      url: event.target.value,
    };
    onOptionsChange({ ...options, jsonData });
  };

  const { jsonData } = options;

  return (
    <div className="gf-form-group">
      <InlineField label="URL" labelWidth={20} tooltip="URL to the Templar version API">
        <Input
          width={40}
          value={jsonData.url || 'http://18.217.218.11/api/templar/version'}
          placeholder="http://18.217.218.11/api/templar/version"
          onChange={onUrlChange}
        />
      </InlineField>
    </div>
  );
}