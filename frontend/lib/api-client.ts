import { backendFetch } from './backend'
import type { components, paths } from '../generated/backend-openapi'

type HttpMethod = 'get' | 'post' | 'put' | 'patch' | 'delete'

type OperationFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = NonNullable<paths[TPath][TMethod]>

type ResponsesFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod>['responses']

type JsonBodyOf<TResponse> = TResponse extends {
  content: infer TContent
}
  ? TContent extends { 'application/json': infer TJson }
    ? TJson
    : never
  : never

type JsonResponseFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  responses: { 200: infer TResponse }
}
  ? JsonBodyOf<TResponse>
  : never

type JsonBodyFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  requestBody?: { content: { 'application/json': infer TBody } }
}
  ? TBody
  : never

type QueryFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  parameters: { query?: infer TQuery }
}
  ? TQuery
  : never

type HTTPErrorResponse = components["schemas"]["HTTPErrorResponse"]

type ExactErrorResponseFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TStatus extends number,
> = TStatus extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>[TStatus]>
  : never

type Fallback4xxErrorFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = '4XX' extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>['4XX']>
  : HTTPErrorResponse

type Fallback5xxErrorFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = '5XX' extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>['5XX']>
  : HTTPErrorResponse

type DefaultErrorFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = 'default' extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>['default']>
  : HTTPErrorResponse

type ExactErrorMembers<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
> = {
  [TStatus in TExactStatuses[number]]: {
    ok: false
    status: TStatus
    error: ExactErrorResponseFor<TPath, TMethod, TStatus>
  }
}[TExactStatuses[number]]

type FallbackErrorMembers<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> =
  | {
      ok: false
      status: '4XX'
      error: Fallback4xxErrorFor<TPath, TMethod>
    }
  | {
      ok: false
      status: '5XX'
      error: Fallback5xxErrorFor<TPath, TMethod>
    }
  | {
      ok: false
      status: 'default'
      error: DefaultErrorFor<TPath, TMethod>
    }

export type EndpointResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[] = [],
> =
  | {
      ok: true
      data: JsonResponseFor<TPath, TMethod>
    }
  | ExactErrorMembers<TPath, TMethod, TExactStatuses>
  | FallbackErrorMembers<TPath, TMethod>

type SyntheticErrorStatus = '4XX' | '5XX' | 'default'

export type ApiSuccess<TValue> = TValue extends { ok: true; data: infer TData }
  ? TData
  : never

export type ApiErrors<TValue> = TValue extends { ok: false; status: infer TStatus; error: infer TError }
  ? { status: TStatus; error: TError }
  : never

function buildQueryString(query: Record<string, unknown> | undefined): string {
  if (!query) return ''
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue
    params.set(key, String(value))
  }
  const serialized = params.toString()
  return serialized ? `?${serialized}` : ''
}

function buildJsonRequestInit(body: unknown, init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers)
  headers.set('Content-Type', 'application/json')
  return {
    ...init,
    headers,
    body: JSON.stringify(body),
  }
}

function buildSyntheticError(code: string, message: string): HTTPErrorResponse {
  return { code, message }
}

function resolveFallbackStatus(httpStatus: number): SyntheticErrorStatus {
  if (httpStatus >= 400 && httpStatus < 500) return '4XX'
  if (httpStatus >= 500 && httpStatus < 600) return '5XX'
  return 'default'
}

function resolveErrorStatus<TExactStatuses extends readonly number[]>(
  httpStatus: number,
  exactErrorStatuses: TExactStatuses,
): TExactStatuses[number] | SyntheticErrorStatus {
  if ((exactErrorStatuses as readonly number[]).includes(httpStatus)) {
    return httpStatus as TExactStatuses[number]
  }
  return resolveFallbackStatus(httpStatus)
}

function buildParsedErrorResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  status: TExactStatuses[number] | SyntheticErrorStatus,
  payload: unknown,
): EndpointResult<TPath, TMethod, TExactStatuses> {
  return {
    ok: false,
    status,
    error: payload as ExactErrorResponseFor<TPath, TMethod, TExactStatuses[number]>
      | Fallback4xxErrorFor<TPath, TMethod>
      | Fallback5xxErrorFor<TPath, TMethod>
      | DefaultErrorFor<TPath, TMethod>,
  } as EndpointResult<TPath, TMethod, TExactStatuses>
}

function buildSyntheticErrorResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  status: SyntheticErrorStatus,
  code: string,
  message: string,
): EndpointResult<TPath, TMethod, TExactStatuses> {
  return {
    ok: false,
    status,
    error: buildSyntheticError(code, message) as Fallback4xxErrorFor<TPath, TMethod>
      | Fallback5xxErrorFor<TPath, TMethod>
      | DefaultErrorFor<TPath, TMethod>,
  } as EndpointResult<TPath, TMethod, TExactStatuses>
}

async function requestEndpointResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  endpoint: TPath,
  method: TMethod,
  exactErrorStatuses: TExactStatuses,
  init?: RequestInit,
  requestPath?: string,
): Promise<EndpointResult<TPath, TMethod, TExactStatuses>> {
  const path = requestPath ?? String(endpoint)

  let response: Response
  try {
    response = await backendFetch(path, {
      method: method.toUpperCase(),
      ...init,
    })
  } catch (error) {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      'default',
      'NETWORK_ERROR',
      error instanceof Error ? error.message : 'Request failed before the server responded.',
    )
  }

  let text = ''
  try {
    text = await response.text()
  } catch (error) {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      resolveFallbackStatus(response.status),
      'RESPONSE_READ_FAILED',
      error instanceof Error ? error.message : 'Failed to read response body.',
    )
  }

  if (response.ok) {
    if (!text) {
      // A 2xx with no body (e.g. a 204 from a DELETE) is a successful void
      // result, not an error. `JsonResponseFor` resolves to `never` for
      // endpoints with no 200 schema, so `null` is a safe placeholder callers
      // ignore in favor of refetching.
      return {
        ok: true,
        data: null as JsonResponseFor<TPath, TMethod>,
      }
    }

    try {
      return {
        ok: true,
        data: JSON.parse(text) as JsonResponseFor<TPath, TMethod>,
      }
    } catch (error) {
      return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
        'default',
        'INVALID_SUCCESS_RESPONSE',
        error instanceof Error ? error.message : 'Server returned invalid JSON.',
      )
    }
  }

  if (!text) {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      resolveFallbackStatus(response.status),
      `HTTP_${response.status}`,
      `${response.status} ${response.statusText || 'Request failed'}`,
    )
  }

  try {
    const payload = JSON.parse(text) as unknown
    return buildParsedErrorResult<TPath, TMethod, TExactStatuses>(
      resolveErrorStatus(response.status, exactErrorStatuses),
      payload,
    )
  } catch {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      resolveFallbackStatus(response.status),
      `HTTP_${response.status}`,
      text,
    )
  }
}

export function makeEndpointClient<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[] = [],
>(
  endpoint: TPath,
  method: TMethod,
  config?: {
    exactErrorStatuses?: TExactStatuses
  },
) {
  const exactErrorStatuses = (config?.exactErrorStatuses ?? []) as TExactStatuses

  return (
    body?: JsonBodyFor<TPath, TMethod>,
    init?: RequestInit,
    requestPath?: string,
  ): Promise<EndpointResult<TPath, TMethod, TExactStatuses>> => {
    const requestInit = body === undefined
      ? init
      : buildJsonRequestInit(body, init)
    return requestEndpointResult(endpoint, method, exactErrorStatuses, requestInit, requestPath)
  }
}

export class ApiClient {
  static getHealth = makeEndpointClient('/health', 'get')

  static getModelDownloadProgress(
    query: QueryFor<'/api/models/download/progress', 'get'>,
  ): Promise<EndpointResult<'/api/models/download/progress', 'get'>> {
    const path = `/api/models/download/progress${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/models/download/progress', 'get', [] as const, undefined, path)
  }

  
  static getLtxRecommendation = makeEndpointClient('/api/models/ltx-recommendation', 'get')

  static getImgGenRecommendation = makeEndpointClient('/api/models/img-gen-recommendation', 'get')

  static getLtxIcLoraRecommendation = makeEndpointClient('/api/models/ltx-ic-lora-recommendation', 'get')

  static getTextEncoderRecommendation = makeEndpointClient('/api/models/text-encoder-recommendation', 'get')

  static startModelDownload = makeEndpointClient('/api/models/download', 'post')

  static deleteModels = makeEndpointClient('/api/models/delete', 'delete')

  static loadModelFromPath = makeEndpointClient('/api/models/load-from-path', 'post')

  static getCheckpointPath(
    query: QueryFor<'/api/models/checkpoint-path', 'get'>,
  ): Promise<EndpointResult<'/api/models/checkpoint-path', 'get'>> {
    const path = `/api/models/checkpoint-path${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/models/checkpoint-path', 'get', [] as const, undefined, path)
  }

  static getRuntimePolicy = makeEndpointClient('/api/runtime-policy', 'get')

  static getSettings = makeEndpointClient('/api/settings', 'get')

  static updateSettings = makeEndpointClient('/api/settings', 'post')

  static suggestGapPrompt = makeEndpointClient('/api/suggest-gap-prompt', 'post', {
    exactErrorStatuses: [401, 403] as const,
  })

  static generateVideo = makeEndpointClient('/api/generate', 'post', {
    exactErrorStatuses: [402] as const,
  })

  static getGenerateVideoModelSpecs = makeEndpointClient('/api/generate/models-specs', 'get')

  static cancelGeneration = makeEndpointClient('/api/generate/cancel', 'post')

  static getGenerationProgress = makeEndpointClient('/api/generation/progress', 'get')

  static generateImage = makeEndpointClient('/api/generate-image', 'post')

  static generateImageEdit = makeEndpointClient('/api/generate-image-edit', 'post')

  static getImageModelSpecs = makeEndpointClient('/api/generate/image-models-specs', 'get')

  // -----------------------------------------------------------------
  // Durable generation queue (video + image).
  //
  // The panel polls `getQueueState` for its whole render; path-parameter
  // endpoints build their URLs manually (makeEndpointClient doesn't
  // substitute path params). DELETE returns 204 (no body), surfaced as a
  // synthetic empty-success error — callers treat it as fire-and-refetch.
  // -----------------------------------------------------------------
  static getQueueState = makeEndpointClient('/api/queue', 'get')

  // In-app LoRA inference registry (Gen Space "Apply LoRA" picker source).
  // Lists the official union IC-LoRA + user-trained adapters + imported LoRAs;
  // the picker popover polls this on open.
  static getLoraInferenceRegistry = makeEndpointClient('/api/lora-inference/registry', 'get')

  // Import an external LoRA file into the in-app library (copies into app
  // storage, tagged with the user-picked variant) and delete an imported one.
  static importLora = makeEndpointClient('/api/lora-inference/import', 'post', {
    exactErrorStatuses: [400] as const,
  })

  static deleteImportedLora(loraId: string) {
    const path = `/api/lora-inference/imported/${encodeURIComponent(loraId)}`
    return requestEndpointResult(
      '/api/lora-inference/imported/{lora_id}',
      'delete',
      [404] as const,
      undefined,
      path,
    )
  }

  // Rename / edit metadata of an imported LoRA (display-only — the weights file
  // is untouched). Returns the updated entry so the picker reflects the change.
  static updateImportedLora(
    loraId: string,
    body: JsonBodyFor<'/api/lora-inference/imported/{lora_id}', 'patch'>,
  ) {
    const path = `/api/lora-inference/imported/${encodeURIComponent(loraId)}`
    return requestEndpointResult(
      '/api/lora-inference/imported/{lora_id}',
      'patch',
      [400, 404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  // Re-derive the per-LoRA system prompt + trigger word for an already-imported
  // LoRA (built-in profile → HuggingFace card → example prompt). Best-effort;
  // returns the profiling outcome so the UI can surface it.
  static reprofileImportedLora(
    loraId: string,
    body: JsonBodyFor<'/api/lora-inference/imported/{lora_id}/reprofile', 'post'>,
  ) {
    const path = `/api/lora-inference/imported/${encodeURIComponent(loraId)}/reprofile`
    return requestEndpointResult(
      '/api/lora-inference/imported/{lora_id}/reprofile',
      'post',
      [404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  // Edit display metadata (name / description) of a user-trained LoRA (backed
  // by the training job). Returns the updated registry entry.
  static updateTrainedLora(
    loraId: string,
    body: JsonBodyFor<'/api/lora-inference/trained/{lora_id}', 'patch'>,
  ) {
    const path = `/api/lora-inference/trained/${encodeURIComponent(loraId)}`
    return requestEndpointResult(
      '/api/lora-inference/trained/{lora_id}',
      'patch',
      [404, 409] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  // Delete a user-trained LoRA (the underlying training job + weights). Rejects
  // while a run is active.
  static deleteTrainedLora(loraId: string) {
    const path = `/api/lora-inference/trained/${encodeURIComponent(loraId)}`
    return requestEndpointResult(
      '/api/lora-inference/trained/{lora_id}',
      'delete',
      [404, 409] as const,
      undefined,
      path,
    )
  }

  // Attach (or replace) a CivitAI-style example image/video on any imported or
  // trained LoRA. `sourcePath` is an absolute path from the Electron file
  // dialog; the backend copies it into app storage and infers the media kind.
  // Returns the updated registry entry so the UI can refresh the thumbnail.
  static setLoraExample(
    loraId: string,
    body: JsonBodyFor<'/api/lora-inference/entries/{lora_id}/example', 'post'>,
  ) {
    const path = `/api/lora-inference/entries/${encodeURIComponent(loraId)}/example`
    return requestEndpointResult(
      '/api/lora-inference/entries/{lora_id}/example',
      'post',
      [400, 404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  // Remove a LoRA's example media (file + stored path). 204 — fire-and-refetch.
  static clearLoraExample(loraId: string) {
    const path = `/api/lora-inference/entries/${encodeURIComponent(loraId)}/example`
    return requestEndpointResult(
      '/api/lora-inference/entries/{lora_id}/example',
      'delete',
      [404] as const,
      undefined,
      path,
    )
  }

  // Backend-relative path for the example-media FileResponse, fed to
  // `useBackendMediaUrl` to render the thumbnail/preview as a blob URL.
  static loraExampleMediaPath(loraId: string): string {
    return `/api/lora-inference/entries/${encodeURIComponent(loraId)}/example-media`
  }

  // Per-LoRA prompt-writing assistant: have Gemini Flash watch the reference
  // video and write a tailored prompt using the LoRA's (editable) system
  // prompt. Gated on a configured Gemini API key (400 when missing).
  static autoPrompt = makeEndpointClient('/api/lora-inference/auto-prompt', 'post', {
    exactErrorStatuses: [400, 404, 409] as const,
  })

  // Persist a user edit of a LoRA's prompt template / trigger word. Passing
  // null for both resets to the auto-generated default. Returns the updated
  // entry so the UI refreshes immediately.
  static updateLoraPromptTemplate(
    loraId: string,
    body: JsonBodyFor<'/api/lora-inference/prompt-template/{lora_id}', 'put'>,
  ) {
    const path = `/api/lora-inference/prompt-template/${encodeURIComponent(loraId)}`
    return requestEndpointResult(
      '/api/lora-inference/prompt-template/{lora_id}',
      'put',
      [404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static enqueueQueueItem = makeEndpointClient('/api/queue/items', 'post', {
    exactErrorStatuses: [422] as const,
  })

  static enqueueQueueBatch = makeEndpointClient('/api/queue/items/batch', 'post', {
    exactErrorStatuses: [422] as const,
  })

  static getQueueItem(itemId: string) {
    const path = `/api/queue/items/${encodeURIComponent(itemId)}`
    return requestEndpointResult('/api/queue/items/{item_id}', 'get', [] as const, undefined, path)
  }

  static updateQueueItem(
    itemId: string,
    body: JsonBodyFor<'/api/queue/items/{item_id}', 'patch'>,
  ) {
    const path = `/api/queue/items/${encodeURIComponent(itemId)}`
    return requestEndpointResult(
      '/api/queue/items/{item_id}',
      'patch',
      [404, 409, 422] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static removeQueueItem(itemId: string) {
    const path = `/api/queue/items/${encodeURIComponent(itemId)}`
    return requestEndpointResult('/api/queue/items/{item_id}', 'delete', [] as const, undefined, path)
  }

  static cancelQueueItem(itemId: string) {
    const path = `/api/queue/items/${encodeURIComponent(itemId)}/cancel`
    return requestEndpointResult(
      '/api/queue/items/{item_id}/cancel',
      'post',
      [404, 409] as const,
      undefined,
      path,
    )
  }

  static reorderQueue = makeEndpointClient('/api/queue/reorder', 'post', {
    exactErrorStatuses: [409] as const,
  })

  static pauseQueue = makeEndpointClient('/api/queue/pause', 'post')

  static resumeQueue = makeEndpointClient('/api/queue/resume', 'post')

  static clearQueueCompleted = makeEndpointClient('/api/queue/clear-completed', 'post')

  static clearQueueFailed = makeEndpointClient('/api/queue/clear-failed', 'post')

  static retake = makeEndpointClient('/api/retake', 'post')

  static startHuggingFaceLogin = makeEndpointClient('/api/auth/huggingface/login', 'post')

  static getHuggingFaceAuthStatus = makeEndpointClient('/api/auth/huggingface/status', 'get')

  static huggingFaceLogout = makeEndpointClient('/api/auth/huggingface/logout', 'post')

  static checkModelAccess = makeEndpointClient('/api/models/check-access', 'post')

  static generateIcLora = makeEndpointClient('/api/ic-lora/generate', 'post')

  static extractIcLoraConditioning = makeEndpointClient('/api/ic-lora/extract-conditioning', 'post')

  // LoRA trainer control plane.
  //
  // The trainer panel polls the three ledgers (datasets / preprocessed /
  // training) via the `list*` getters; mutations refresh on success.
  // Path-parameter endpoints build their URLs manually because
  // `makeEndpointClient` doesn't substitute path params. The DELETE
  // routes return 204 (no body), which the typed client surfaces as a
  // synthetic "empty success" error — callers treat those as fire-and-
  // refetch, identical to the queue's remove flow.
  static listLoraDatasets(
    query: QueryFor<'/api/lora/datasets', 'get'> = {},
  ): Promise<EndpointResult<'/api/lora/datasets', 'get'>> {
    const path = `/api/lora/datasets${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/lora/datasets', 'get', [] as const, undefined, path)
  }

  static createLoraDataset = makeEndpointClient('/api/lora/datasets', 'post')

  static updateLoraDataset(
    datasetId: string,
    body: JsonBodyFor<'/api/lora/datasets/{dataset_id}', 'patch'>,
  ) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}`
    return requestEndpointResult(
      '/api/lora/datasets/{dataset_id}',
      'patch',
      [404, 409] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static deleteLoraDataset(datasetId: string) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}`
    return requestEndpointResult('/api/lora/datasets/{dataset_id}', 'delete', [] as const, undefined, path)
  }

  static archiveLoraDataset(datasetId: string) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/archive`
    return requestEndpointResult('/api/lora/datasets/{dataset_id}/archive', 'post', [404, 409] as const, undefined, path)
  }

  static unarchiveLoraDataset(datasetId: string) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/unarchive`
    return requestEndpointResult('/api/lora/datasets/{dataset_id}/unarchive', 'post', [404] as const, undefined, path)
  }

  static uploadLoraDataset(
    datasetId: string,
    body: JsonBodyFor<'/api/lora/datasets/{dataset_id}/upload', 'post'>,
  ) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/upload`
    return requestEndpointResult(
      '/api/lora/datasets/{dataset_id}/upload',
      'post',
      [404, 409] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static cancelLoraUpload(datasetId: string) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/cancel`
    return requestEndpointResult('/api/lora/datasets/{dataset_id}/cancel', 'post', [404, 409] as const, undefined, path)
  }

  static renameLoraDataset(datasetId: string, body: JsonBodyFor<'/api/lora/datasets/{dataset_id}/rename', 'post'>) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/rename`
    return requestEndpointResult('/api/lora/datasets/{dataset_id}/rename', 'post', [404, 409] as const, buildJsonRequestInit(body), path)
  }

  static createLoraFolder(body: JsonBodyFor<'/api/lora/folders', 'post'>) {
    return requestEndpointResult('/api/lora/folders', 'post', [404, 409] as const, buildJsonRequestInit(body))
  }

  static renameLoraFolder(folderId: string, body: JsonBodyFor<'/api/lora/folders/{folder_id}', 'patch'>) {
    const path = `/api/lora/folders/${encodeURIComponent(folderId)}`
    return requestEndpointResult('/api/lora/folders/{folder_id}', 'patch', [404, 409] as const, buildJsonRequestInit(body), path)
  }

  static moveLoraFolder(folderId: string, body: JsonBodyFor<'/api/lora/folders/{folder_id}/move', 'post'>) {
    const path = `/api/lora/folders/${encodeURIComponent(folderId)}/move`
    return requestEndpointResult('/api/lora/folders/{folder_id}/move', 'post', [404, 409] as const, buildJsonRequestInit(body), path)
  }

  static deleteLoraFolder(folderId: string, recursive: boolean) {
    const path = `/api/lora/folders/${encodeURIComponent(folderId)}?recursive=${recursive}`
    return requestEndpointResult('/api/lora/folders/{folder_id}', 'delete', [404, 409] as const, undefined, path)
  }

  static moveLoraDataset(datasetId: string, body: JsonBodyFor<'/api/lora/datasets/{dataset_id}/move', 'post'>) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/move`
    return requestEndpointResult('/api/lora/datasets/{dataset_id}/move', 'post', [404, 409] as const, buildJsonRequestInit(body), path)
  }

  static exportLoraDataset(
    datasetId: string,
    body: JsonBodyFor<'/api/lora/datasets/{dataset_id}/export', 'post'>,
  ) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/export`
    return requestEndpointResult(
      '/api/lora/datasets/{dataset_id}/export',
      'post',
      [400, 404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static importLoraDataset = makeEndpointClient('/api/lora/datasets/import', 'post', {
    exactErrorStatuses: [400] as const,
  })

  static publishLoraPreview(
    trainingId: string,
    body: JsonBodyFor<'/api/lora/training/{training_id}/publish/preview', 'post'>,
  ) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/publish/preview`
    return requestEndpointResult(
      '/api/lora/training/{training_id}/publish/preview',
      'post',
      [400, 404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static publishLoraExport(
    trainingId: string,
    body: JsonBodyFor<'/api/lora/training/{training_id}/publish/export', 'post'>,
  ) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/publish/export`
    return requestEndpointResult(
      '/api/lora/training/{training_id}/publish/export',
      'post',
      [400, 404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static listLoraPreprocessed = makeEndpointClient('/api/lora/preprocessed', 'get')

  static startLoraPreprocessing = makeEndpointClient('/api/lora/preprocessed', 'post', {
    exactErrorStatuses: [404, 409] as const,
  })

  static cancelLoraPreprocessing(preprocessedId: string) {
    const path = `/api/lora/preprocessed/${encodeURIComponent(preprocessedId)}/cancel`
    return requestEndpointResult('/api/lora/preprocessed/{preprocessed_id}/cancel', 'post', [404, 409] as const, undefined, path)
  }

  static resumeLoraPreprocessing(preprocessedId: string) {
    const path = `/api/lora/preprocessed/${encodeURIComponent(preprocessedId)}/resume`
    return requestEndpointResult('/api/lora/preprocessed/{preprocessed_id}/resume', 'post', [404, 409] as const, undefined, path)
  }

  static resetLoraPreprocessing(preprocessedId: string) {
    const path = `/api/lora/preprocessed/${encodeURIComponent(preprocessedId)}/reset`
    return requestEndpointResult('/api/lora/preprocessed/{preprocessed_id}/reset', 'post', [404, 409] as const, undefined, path)
  }

  static deleteLoraPreprocessed(preprocessedId: string) {
    const path = `/api/lora/preprocessed/${encodeURIComponent(preprocessedId)}`
    return requestEndpointResult('/api/lora/preprocessed/{preprocessed_id}', 'delete', [] as const, undefined, path)
  }

  static listLoraTraining(
    query: QueryFor<'/api/lora/training', 'get'> = {},
  ): Promise<EndpointResult<'/api/lora/training', 'get'>> {
    const path = `/api/lora/training${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/lora/training', 'get', [] as const, undefined, path)
  }

  static startLoraTraining = makeEndpointClient('/api/lora/training', 'post', {
    exactErrorStatuses: [404, 409] as const,
  })

  static startLoraTrainingPipeline = makeEndpointClient('/api/lora/training-pipeline', 'post', {
    exactErrorStatuses: [404, 409, 422] as const,
  })

  static cancelLoraTraining(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/cancel`
    return requestEndpointResult('/api/lora/training/{training_id}/cancel', 'post', [404, 409] as const, undefined, path)
  }

  static deleteLoraTraining(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}`
    return requestEndpointResult('/api/lora/training/{training_id}', 'delete', [] as const, undefined, path)
  }

  static archiveLoraTraining(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/archive`
    return requestEndpointResult('/api/lora/training/{training_id}/archive', 'post', [404, 409] as const, undefined, path)
  }

  static unarchiveLoraTraining(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/unarchive`
    return requestEndpointResult('/api/lora/training/{training_id}/unarchive', 'post', [404] as const, undefined, path)
  }

  static retryLoraTrainingDownload(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/retry-download`
    return requestEndpointResult('/api/lora/training/{training_id}/retry-download', 'post', [404, 409] as const, undefined, path)
  }

  static resumeLoraTraining(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/resume`
    return requestEndpointResult('/api/lora/training/{training_id}/resume', 'post', [404, 409] as const, undefined, path)
  }

  static resetLoraTraining(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/reset`
    return requestEndpointResult('/api/lora/training/{training_id}/reset', 'post', [404, 409] as const, undefined, path)
  }

  static getLoraTrainingLogs(trainingId: string) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/logs`
    return requestEndpointResult('/api/lora/training/{training_id}/logs', 'get', [] as const, undefined, path)
  }

  static testLoraConnection = makeEndpointClient('/api/lora/test-connection', 'post')

  // Read-only capability probe: whether local (WSL2) training is possible on
  // this machine. Never errors — an unavailable setup is reported as
  // `eligible=false` with a `reason`.
  static getLoraLocalEligibility = makeEndpointClient('/api/lora/local-eligibility', 'get')

  static connectRunpod = makeEndpointClient('/api/lora/runpod/connect', 'post')

  static estimateRunpodTraining = makeEndpointClient('/api/lora/training/estimate', 'post')

  static reselectLoraDatasetRunpod(
    datasetId: string,
    body: JsonBodyFor<'/api/lora/datasets/{dataset_id}/reselect-runpod', 'post'>,
  ) {
    const path = `/api/lora/datasets/${encodeURIComponent(datasetId)}/reselect-runpod`
    return requestEndpointResult(
      '/api/lora/datasets/{dataset_id}/reselect-runpod',
      'post',
      [404, 409] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static reselectLoraTrainingRunpod(
    trainingId: string,
    body: JsonBodyFor<'/api/lora/training/{training_id}/reselect-runpod', 'post'>,
  ) {
    const path = `/api/lora/training/${encodeURIComponent(trainingId)}/reselect-runpod`
    return requestEndpointResult(
      '/api/lora/training/{training_id}/reselect-runpod',
      'post',
      [404, 409] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static createRunpodVolume = makeEndpointClient('/api/lora/runpod/volumes/create', 'post', {
    exactErrorStatuses: [409] as const,
  })

  static selectRunpodVolume = makeEndpointClient('/api/lora/runpod/volumes/select', 'post', {
    exactErrorStatuses: [409] as const,
  })

  static disableRunpodCache = makeEndpointClient('/api/lora/runpod/cache/disable', 'post')

  static relocateRunpodVolume = makeEndpointClient('/api/lora/runpod/volumes/relocate', 'post', {
    exactErrorStatuses: [409] as const,
  })

  static deleteRunpodVolume(volumeId: string) {
    const path = `/api/lora/runpod/volumes/${encodeURIComponent(volumeId)}`
    return requestEndpointResult('/api/lora/runpod/volumes/{volume_id}', 'delete', [409] as const, undefined, path)
  }

  // Trainer compute panel: list every pod on the account (standalone, no GPU
  // discovery) and pause/resume individual pods so stray compute can't keep
  // billing. Terminate stays available too. Path-param endpoints build their
  // URLs manually (makeEndpointClient doesn't substitute path params).
  static listRunpodPods = makeEndpointClient('/api/lora/runpod/pods', 'get')

  static terminateRunpodPod(podId: string) {
    const path = `/api/lora/runpod/pods/${encodeURIComponent(podId)}/terminate`
    return requestEndpointResult('/api/lora/runpod/pods/{pod_id}/terminate', 'post', [] as const, undefined, path)
  }

  static stopRunpodPod(podId: string) {
    const path = `/api/lora/runpod/pods/${encodeURIComponent(podId)}/stop`
    return requestEndpointResult('/api/lora/runpod/pods/{pod_id}/stop', 'post', [] as const, undefined, path)
  }

  static resumeRunpodPod(podId: string) {
    const path = `/api/lora/runpod/pods/${encodeURIComponent(podId)}/resume`
    return requestEndpointResult('/api/lora/runpod/pods/{pod_id}/resume', 'post', [] as const, undefined, path)
  }

  static keepRunpodPodAlive(podId: string, minutes = 30) {
    const path = `/api/lora/runpod/pods/${encodeURIComponent(podId)}/keep-alive`
    return requestEndpointResult(
      '/api/lora/runpod/pods/{pod_id}/keep-alive',
      'post',
      [404, 409] as const,
      buildJsonRequestInit({ minutes }),
      path,
    )
  }

  static captionLoraClip = makeEndpointClient('/api/lora/caption-clip', 'post', {
    exactErrorStatuses: [400, 413, 502, 504] as const,
  })

  static probeLoraClip = makeEndpointClient('/api/lora/probe-clip', 'post', {
    exactErrorStatuses: [400, 422, 500, 504] as const,
  })

  static applyLoraClipEdits = makeEndpointClient('/api/lora/apply-edits', 'post', {
    exactErrorStatuses: [400, 422, 500, 504] as const,
  })

  static splitLoraScenes = makeEndpointClient('/api/lora/scene-split', 'post', {
    exactErrorStatuses: [400, 422, 500, 504] as const,
  })

  static editLoraFrame = makeEndpointClient('/api/lora/edit-frame', 'post', {
    exactErrorStatuses: [400, 422, 500, 502, 504] as const,
  })

  static extractMediaFrame = makeEndpointClient('/api/media/extract-frame', 'post', {
    exactErrorStatuses: [422] as const,
  })

  static animateLoraFrame = makeEndpointClient('/api/lora/animate-frame', 'post', {
    exactErrorStatuses: [400, 422, 500, 502, 504] as const,
  })

  static restyleLoraClip = makeEndpointClient('/api/lora/restyle-clip', 'post', {
    exactErrorStatuses: [400, 422, 500, 502, 504] as const,
  })

  static motionEditLoraClip = makeEndpointClient('/api/lora/motion-edit', 'post', {
    exactErrorStatuses: [400, 422, 500, 502, 504] as const,
  })

  static searchPexels = makeEndpointClient('/api/lora/pexels/search', 'post', {
    exactErrorStatuses: [400, 422, 429, 500, 502, 504] as const,
  })

  static downloadPexels = makeEndpointClient('/api/lora/pexels/download', 'post', {
    exactErrorStatuses: [400, 422, 429, 500, 502, 504] as const,
  })

  static enqueueLoraClipJobs = makeEndpointClient('/api/lora/clip-jobs', 'post', {
    exactErrorStatuses: [400, 422] as const,
  })

  static listLoraClipJobs = makeEndpointClient('/api/lora/clip-jobs', 'get')

  static createLoraDerivation = makeEndpointClient('/api/lora/derivations', 'post', {
    exactErrorStatuses: [400, 422] as const,
  })

  static listLoraDerivations = makeEndpointClient('/api/lora/derivations', 'get')

  static cancelAllLoraDerivations = makeEndpointClient('/api/lora/derivations/cancel-all', 'post')

  static cancelLoraDerivation(jobId: string) {
    const path = `/api/lora/derivations/${encodeURIComponent(jobId)}/cancel`
    return requestEndpointResult('/api/lora/derivations/{job_id}/cancel', 'post', [404] as const, undefined, path)
  }

  static retryLoraDerivation(jobId: string) {
    const path = `/api/lora/derivations/${encodeURIComponent(jobId)}/retry`
    return requestEndpointResult('/api/lora/derivations/{job_id}/retry', 'post', [409] as const, undefined, path)
  }

  static approveLoraDerivation(jobId: string) {
    const path = `/api/lora/derivations/${encodeURIComponent(jobId)}/approve`
    return requestEndpointResult('/api/lora/derivations/{job_id}/approve', 'post', [409] as const, undefined, path)
  }

  static regenerateLoraDerivationEdit(jobId: string, editPrompt?: string) {
    const path = `/api/lora/derivations/${encodeURIComponent(jobId)}/regenerate-edit`
    return requestEndpointResult(
      '/api/lora/derivations/{job_id}/regenerate-edit',
      'post',
      [409] as const,
      buildJsonRequestInit({ editPrompt: editPrompt ?? null }),
      path,
    )
  }

  static dismissLoraDerivation(jobId: string) {
    const path = `/api/lora/derivations/${encodeURIComponent(jobId)}/dismiss`
    return requestEndpointResult('/api/lora/derivations/{job_id}/dismiss', 'post', [] as const, undefined, path)
  }

  static listLoraProfiles = makeEndpointClient('/api/lora/profiles', 'get')

  static createLoraProfile = makeEndpointClient('/api/lora/profiles', 'post', {
    exactErrorStatuses: [422] as const,
  })

  static updateLoraProfile(
    profileId: string,
    body: JsonBodyFor<'/api/lora/profiles/{profile_id}', 'patch'>,
  ) {
    const path = `/api/lora/profiles/${encodeURIComponent(profileId)}`
    return requestEndpointResult(
      '/api/lora/profiles/{profile_id}',
      'patch',
      [404] as const,
      buildJsonRequestInit(body),
      path,
    )
  }

  static deleteLoraProfile(profileId: string) {
    const path = `/api/lora/profiles/${encodeURIComponent(profileId)}`
    return requestEndpointResult('/api/lora/profiles/{profile_id}', 'delete', [404] as const, undefined, path)
  }
}

type ApiClientMethodName = keyof typeof ApiClient

export type ApiRequestBodyOf<TMethod extends ApiClientMethodName> = (typeof ApiClient)[TMethod] extends (
  body?: infer TBody,
  ...args: any[]
) => Promise<any>
  ? TBody
  : never

export type ApiSuccessOf<TMethod extends ApiClientMethodName> = (typeof ApiClient)[TMethod] extends (...args: any[]) => Promise<any>
  ? ApiSuccess<Awaited<ReturnType<(typeof ApiClient)[TMethod]>>>
  : never

export type ApiErrorsOf<TMethod extends ApiClientMethodName> = (typeof ApiClient)[TMethod] extends (...args: any[]) => Promise<any>
  ? ApiErrors<Awaited<ReturnType<(typeof ApiClient)[TMethod]>>>
  : never

// `updateLoraProfile` takes (id, body), so `ApiRequestBodyOf` can't infer the
// body (it reads the first arg). Expose the PATCH body type directly instead.
export type UpdateLoraProfileBody = JsonBodyFor<'/api/lora/profiles/{profile_id}', 'patch'>

// Publish methods also take (id, body); expose their bodies directly too.
export type PublishLoraPreviewBody = JsonBodyFor<'/api/lora/training/{training_id}/publish/preview', 'post'>
export type PublishLoraExportBody = JsonBodyFor<'/api/lora/training/{training_id}/publish/export', 'post'>
