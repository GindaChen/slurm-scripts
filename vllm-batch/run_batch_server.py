"""
mkdir -p output_tmp
python run_batch_server.py --disable-log-requests --multi-step-stream-outputs 32 \
--model meta-llama/Meta-Llama-3-8B-Instruct \
-i data/input.jsonl \
-o output_tmp/output.jsonl \
--output-tmp-dir output_tmp/

usage: run_batch.py [-h] -i INPUT_FILE -o OUTPUT_FILE [--response-role RESPONSE_ROLE] [--model MODEL] [--task {auto,generate,embedding,embed,classify,score,reward}] [--tokenizer TOKENIZER] [--skip-tokenizer-init] [--revision REVISION] [--code-revision CODE_REVISION]
                    [--tokenizer-revision TOKENIZER_REVISION] [--tokenizer-mode {auto,slow,mistral}] [--trust-remote-code] [--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH] [--download-dir DOWNLOAD_DIR]
                    [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer}] [--config-format {auto,hf,mistral}] [--dtype {auto,half,float16,bfloat16,float,float32}]
                    [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}] [--max-model-len MAX_MODEL_LEN] [--guided-decoding-backend {outlines,lm-format-enforcer,xgrammar}] [--logits-processor-pattern LOGITS_PROCESSOR_PATTERN] [--model-impl {auto,vllm,transformers}]
                    [--distributed-executor-backend {ray,mp,uni,external_launcher}] [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS] [--ray-workers-use-nsight]
                    [--block-size {8,16,32,64,128}] [--enable-prefix-caching | --no-enable-prefix-caching] [--disable-sliding-window] [--use-v2-block-manager] [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS] [--seed SEED] [--swap-space SWAP_SPACE]
                    [--cpu-offload-gb CPU_OFFLOAD_GB] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION] [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE] [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS] [--max-num-seqs MAX_NUM_SEQS] [--max-logprobs MAX_LOGPROBS]
                    [--disable-log-stats] [--quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}]
                    [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA] [--hf-overrides HF_OVERRIDES] [--enforce-eager] [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE] [--disable-custom-all-reduce] [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
                    [--tokenizer-pool-type TOKENIZER_POOL_TYPE] [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG] [--limit-mm-per-prompt LIMIT_MM_PER_PROMPT] [--mm-processor-kwargs MM_PROCESSOR_KWARGS] [--disable-mm-preprocessor-cache] [--enable-lora]
                    [--enable-lora-bias] [--max-loras MAX_LORAS] [--max-lora-rank MAX_LORA_RANK] [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE] [--lora-dtype {auto,float16,bfloat16}] [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
                    [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras] [--enable-prompt-adapter] [--max-prompt-adapters MAX_PROMPT_ADAPTERS] [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN] [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu,hpu}]
                    [--num-scheduler-steps NUM_SCHEDULER_STEPS] [--multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]] [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR] [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]] [--speculative-model SPECULATIVE_MODEL]
                    [--speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}]
                    [--num-speculative-tokens NUM_SPECULATIVE_TOKENS] [--speculative-disable-mqa-scorer] [--speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE] [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
                    [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE] [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX] [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
                    [--spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}] [--typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD]
                    [--typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA] [--disable-logprobs-during-spec-decoding [DISABLE_LOGPROBS_DURING_SPEC_DECODING]] [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
                    [--ignore-patterns IGNORE_PATTERNS] [--preemption-mode PREEMPTION_MODE] [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]] [--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH] [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
                    [--collect-detailed-traces COLLECT_DETAILED_TRACES] [--disable-async-output-proc] [--scheduling-policy {fcfs,priority}] [--override-neuron-config OVERRIDE_NEURON_CONFIG] [--override-pooler-config OVERRIDE_POOLER_CONFIG]
                    [--compilation-config COMPILATION_CONFIG] [--kv-transfer-config KV_TRANSFER_CONFIG] [--worker-cls WORKER_CLS] [--generation-config GENERATION_CONFIG] [--override-generation-config OVERRIDE_GENERATION_CONFIG] [--enable-sleep-mode]
                    [--calculate-kv-scales] [--disable-log-requests] [--max-log-len MAX_LOG_LEN] [--enable-metrics] [--url URL] [--port PORT] [--enable-prompt-tokens-details]
"""
from datetime import datetime
import importlib.util
import sys
import asyncio
import tempfile
from http import HTTPStatus
from io import StringIO
from typing import Awaitable, Callable, List, Optional

import aiohttp
import torch
from prometheus_client import start_http_server
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger, logger
# yapf: disable
from vllm.entrypoints.openai.protocol import (BatchRequestInput,
                                              BatchRequestOutput,
                                              BatchResponseData,
                                              ChatCompletionResponse,
                                              EmbeddingResponse, ErrorResponse,
                                              ScoreResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_score import OpenAIServingScores
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.version import __version__ as VLLM_VERSION


def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible batch runner.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help=
        "The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.")
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser.add_argument(
        "-o",
        "--output-file",
        required=False,
        type=str,
        default=f"output_{now}.jsonl",
        help="The path or url to a single output file. Currently supports "
        "local file paths, or web (http or https) urls. If a URL is specified,"
        " the file should be available via HTTP PUT.")
    
    
    parser.add_argument(
        "--output-tmp-dir",
        type=str,
        default=f"output_tmp_{now}",
        help="The directory to store the output file before uploading it "
        "to the output URL.",
    )
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=True`.")

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')

    parser.add_argument("--enable-metrics",
                        action="store_true",
                        help="Enable Prometheus metrics")
    parser.add_argument(
        "--url",
        type=str,
        default="0.0.0.0",
        help="URL to the Prometheus metrics server "
        "(only needed if enable-metrics is set).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the Prometheus metrics server "
        "(only needed if enable-metrics is set).",
    )
    parser.add_argument(
        "--enable-prompt-tokens-details",
        action='store_true',
        default=False,
        help="If set to True, enable prompt_tokens_details in usage.")
    
    parser.add_argument("--custome-request-script",
                        type=str,
                        default=None,
                        help="Customize the request script.")

    parser.add_argument("--checkpoint-frequency",
                        type=int,
                        default=None,
                        help="The frequency of checkpoint.")
    return parser.parse_args()


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


class BatchProgressTracker:

    def __init__(self):
        self._total = 0
        self._pbar: Optional[tqdm] = None

    def submitted(self):
        self._total += 1

    def completed(self):
        if self._pbar:
            self._pbar.update()

    def pbar(self) -> tqdm:
        enable_tqdm = not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0
        self._pbar = tqdm(total=self._total,
                          unit="req",
                          desc="Running batch",
                          mininterval=5,
                          disable=not enable_tqdm,
                          bar_format=_BAR_FORMAT)
        return self._pbar


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, encoding="utf-8") as f:
            return f.read()


async def write_local_file(output_path: str,
                           batch_outputs: List[BatchRequestOutput]) -> None:
    """
    Write the responses to a local file.
    output_path: The path to write the responses to.
    batch_outputs: The list of batch outputs to write.
    """
    # We should make this async, but as long as run_batch runs as a
    # standalone program, blocking the event loop won't effect performance.
    with open(output_path, "w", encoding="utf-8") as f:
        for o in batch_outputs:
            print(o.model_dump_json(), file=f)


async def upload_data(output_url: str, data_or_file: str,
                      from_file: bool) -> None:
    """
    Upload a local file to a URL.
    output_url: The URL to upload the file to.
    data_or_file: Either the data to upload or the path to the file to upload.
    from_file: If True, data_or_file is the path to the file to upload.
    """
    # Timeout is a common issue when uploading large files.
    # We retry max_retries times before giving up.
    max_retries = 5
    # Number of seconds to wait before retrying.
    delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            # We increase the timeout to 1000 seconds to allow
            # for large files (default is 300).
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
                    total=1000)) as session:
                if from_file:
                    with open(data_or_file, "rb") as file:
                        async with session.put(output_url,
                                               data=file) as response:
                            if response.status != 200:
                                raise Exception(f"Failed to upload file.\n"
                                                f"Status: {response.status}\n"
                                                f"Response: {response.text()}")
                else:
                    async with session.put(output_url,
                                           data=data_or_file) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to upload data.\n"
                                            f"Status: {response.status}\n"
                                            f"Response: {response.text()}")

        except Exception as e:
            if attempt < max_retries:
                logger.error(
                    f"Failed to upload data (attempt {attempt}). "
                    f"Error message: {str(e)}.\nRetrying in {delay} seconds..."
                )
                await asyncio.sleep(delay)
            else:
                raise Exception(f"Failed to upload data (attempt {attempt}). "
                                f"Error message: {str(e)}.") from e


async def write_file(path_or_url: str, batch_outputs: List[BatchRequestOutput],
                     output_tmp_dir: str) -> None:
    """
    Write batch_outputs to a file or upload to a URL.
    path_or_url: The path or URL to write batch_outputs to.
    batch_outputs: The list of batch outputs to write.
    output_tmp_dir: The directory to store the output file before uploading it
    to the output URL.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        if output_tmp_dir is None:
            logger.info("Writing outputs to memory buffer")
            output_buffer = StringIO()
            for o in batch_outputs:
                print(o.model_dump_json(), file=output_buffer)
            output_buffer.seek(0)
            logger.info("Uploading outputs to %s", path_or_url)
            await upload_data(
                path_or_url,
                output_buffer.read().strip().encode("utf-8"),
                from_file=False,
            )
        else:
            # Write responses to a temporary file and then upload it to the URL.
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=output_tmp_dir,
                    prefix="tmp_batch_output_",
                    suffix=".jsonl",
            ) as f:
                logger.info("Writing outputs to temporary local file %s",
                            f.name)
                await write_local_file(f.name, batch_outputs)
                logger.info("Uploading outputs to %s", path_or_url)
                await upload_data(path_or_url, f.name, from_file=True)
    else:
        logger.info("Writing outputs to local file %s", path_or_url)
        await write_local_file(path_or_url, batch_outputs)


def make_error_request_output(request: BatchRequestInput,
                              error_msg: str) -> BatchRequestOutput:
    batch_output = BatchRequestOutput(
        id=f"vllm-{random_uuid()}",
        custom_id=request.custom_id,
        response=BatchResponseData(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id=f"vllm-batch-{random_uuid()}",
        ),
        error=error_msg,
    )
    return batch_output


async def make_async_error_request_output(
        request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    return make_error_request_output(request, error_msg)


async def run_request(serving_engine_func: Callable,
                      request: BatchRequestInput,
                      tracker: BatchProgressTracker) -> BatchRequestOutput:
    response = await serving_engine_func(request.body)

    if isinstance(response,
                  (ChatCompletionResponse, EmbeddingResponse, ScoreResponse)):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                body=response, request_id=f"vllm-batch-{random_uuid()}"),
            error=None,
        )
    elif isinstance(response, ErrorResponse):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                status_code=response.code,
                request_id=f"vllm-batch-{random_uuid()}"),
            error=response,
        )
    else:
        batch_output = make_error_request_output(
            request, error_msg="Request must not be sent in stream mode")

    tracker.completed()
    return batch_output


async def prepare_requests_default(input_file: str):
    requests = []
    for request_json in (await read_file(input_file)).strip().split("\n"):
        # Skip empty lines.
        request_json = request_json.strip()
        if not request_json:
            continue

        request = BatchRequestInput.model_validate_json(request_json)
        requests.append(request)
    return requests


async def prepare_requests_custome(input_file: str, custome_request_script: str):
    # Load the custom request script module
    spec = importlib.util.spec_from_file_location(
        "custom_requests", 
        custome_request_script
    )

    if spec is None:
        raise ImportError(f"Could not load spec from {custome_request_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_requests"] = module
    spec.loader.exec_module(module)

    # Get requests from the custom module
    if not hasattr(module, "get_requests"):
        raise AttributeError(f"Custom script {custome_request_script} must define get_requests() function")
    
    if asyncio.iscoroutinefunction(module.get_requests):
        requests = await module.get_requests(input_file)
    else:
        requests = module.get_requests(input_file)
    return requests


async def prepare_requests(input_file: str, custome_request_script: str):
    if not custome_request_script:
        return await prepare_requests_default(input_file)
    else:
        return await prepare_requests_custome(input_file, custome_request_script)

async def async_as_completed(tasks, every=1):
    """
    Yield lists of completed results from tasks as soon as each list reaches `every` items.
    
    Args:
        tasks (Iterable[asyncio.Task]): An iterable of asyncio tasks or coroutines.
        every (int): The batch size to yield.
    
    Yields:
        List: A list of completed results.
    """
    batch = []
    for completed in asyncio.as_completed(tasks):
        result = await completed
        batch.append(result)
        if len(batch) >= every:
            yield batch
            batch = []
    # Yield any remaining results that didn't form a full batch.
    if batch:
        yield batch


async def main(args):
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_BATCH_RUNNER)

    model_config = await engine.get_model_config()
    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    # Create the openai serving objects.
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=None,
        prompt_adapters=None,
    )
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=None,
        chat_template_content_format="auto",
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None
    openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
        chat_template=None,
        chat_template_content_format="auto",
    ) if model_config.task == "embed" else None
    openai_serving_scores = (OpenAIServingScores(
        engine,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
    ) if model_config.task == "score" else None)

    tracker = BatchProgressTracker()
    logger.info("Reading batch from %s...", args.input_file)

    # Submit all requests in the file to the engine "concurrently".
    response_futures: List[Awaitable[BatchRequestOutput]] = []
    
    # custome-request-script
    requests = await prepare_requests(args.input_file, args.custome_request_script)

    for request in requests:
        # Determine the type of request and run it.
        if request.url == "/v1/chat/completions":
            handler_fn = (None if openai_serving_chat is None else
                          openai_serving_chat.create_chat_completion)
            if handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg=
                        "The model does not support Chat Completions API",
                    ))
                continue

            response_futures.append(run_request(handler_fn, request, tracker))
            tracker.submitted()
        elif request.url == "/v1/embeddings":
            handler_fn = (None if openai_serving_embedding is None else
                          openai_serving_embedding.create_embedding)
            if handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Embeddings API",
                    ))
                continue

            response_futures.append(run_request(handler_fn, request, tracker))
            tracker.submitted()
        elif request.url == "/v1/score":
            handler_fn = (None if openai_serving_scores is None else
                          openai_serving_scores.create_score)
            if handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Scores API",
                    ))
                continue

            response_futures.append(run_request(handler_fn, request, tracker))
            tracker.submitted()
        else:
            response_futures.append(
                make_async_error_request_output(
                    request,
                    error_msg=
                    "Only /v1/chat/completions, /v1/embeddings, and /v1/score "
                    "are supported in the batch endpoint.",
                ))

    # with tracker.pbar():
    #     responses = await asyncio.gather(*response_futures)
    checkpoint_frequency = args.checkpoint_frequency
    if checkpoint_frequency is None:
        checkpoint_frequency = len(response_futures)

    responses = []
    with tracker.pbar():
        idx = 0
        async for batch in async_as_completed(response_futures, every=checkpoint_frequency):
            responses.extend(batch)
            output_file = f"{args.output_file}.{idx}.tmp"
            await write_file(output_file, batch, args.output_tmp_dir)
            idx += 1

    await write_file(args.output_file, responses, args.output_tmp_dir)


if __name__ == "__main__":
    args = parse_args()

    logger.info("vLLM batch processing API version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    # Start the Prometheus metrics server. LLMEngine uses the Prometheus client
    # to publish metrics at the /metrics endpoint.
    if args.enable_metrics:
        logger.info("Prometheus metrics enabled")
        start_http_server(port=args.port, addr=args.url)
    else:
        logger.info("Prometheus metrics disabled")

    asyncio.run(main(args))