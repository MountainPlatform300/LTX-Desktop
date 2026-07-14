import type { components } from '../generated/backend-openapi'

export type ImageModelSpecsResponse = components['schemas']['GenerateImageModelsSpecsResponse']
export type ImageModelSpec = components['schemas']['ImageModelSpecApi']
export type ImageModelInferenceStatus = ImageModelSpec['inference_status']
