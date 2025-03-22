import {
  DataQueryRequest,
  DataQueryResponse,
  DataSourceApi,
  DataSourceInstanceSettings,
  FieldType,
  MutableDataFrame,
} from '@grafana/data';
import { getBackendSrv } from '@grafana/runtime';
import { lastValueFrom } from 'rxjs';
import { TemplarVersionDataSourceOptions, TemplarVersionQuery } from './types';

export class TemplarVersionDataSource extends DataSourceApi<TemplarVersionQuery, TemplarVersionDataSourceOptions> {
  url: string;

  constructor(instanceSettings: DataSourceInstanceSettings<TemplarVersionDataSourceOptions>) {
    super(instanceSettings);
    this.url = instanceSettings.jsonData.url || 'http://18.217.218.11/api/templar/version';
  }

  async query(options: DataQueryRequest<TemplarVersionQuery>): Promise<DataQueryResponse> {
    const { range } = options;
    const from = range!.from.valueOf();
    const to = range!.to.valueOf();

    // Return a frame with the version information
    try {
      const response = await lastValueFrom(
        getBackendSrv().fetch({
          url: this.url,
          method: 'GET',
        })
      );

      if (response.status === 200) {
        const data = response.data as { version: string };

        // Create a data frame with the version
        const frame = new MutableDataFrame({
          refId: options.targets[0].refId,
          fields: [
            { name: 'time', type: FieldType.time, values: [to] },
            { name: 'version', type: FieldType.string, values: [data.version] }
          ],
        });

        return { data: [frame] };
      } else {
        throw new Error(`Failed to fetch version: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error fetching version:', error);
      return { data: [] };
    }
  }

  async testDatasource() {
    try {
      const response = await lastValueFrom(
        getBackendSrv().fetch({
          url: this.url,
          method: 'GET',
        })
      );

      if (response.status === 200) {
        return {
          status: 'success',
          message: 'Templar version API is working',
        };
      } else {
        return {
          status: 'error',
          message: `Failed to connect to the API: ${response.statusText}`,
        };
      }
    } catch (error) {
      return {
        status: 'error',
        message: `Failed to connect to the API: ${error}`,
      };
    }
  }
}
