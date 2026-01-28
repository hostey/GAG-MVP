# services/simulation_service.py
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from core.engine.simulation_engine import SimulationOrchestrator, SimulationResult
from core.utils.logger import get_logger
from services.cache_service import CacheService

logger = get_logger(__name__)


class SimulationService:
    """Service layer for managing simulation operations."""

    def __init__(self, max_workers: int = 4):
        self.orchestrator = SimulationOrchestrator(max_concurrent=max_workers)
        self.cache_service = CacheService()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Track active simulations
        self.active_simulations: Dict[str, Dict] = {}
        self.simulation_history: List[Dict] = []

        logger.info(f"Initialized SimulationService with {max_workers} workers")

    async def run_simulation(
            self,
            parameters: Dict[str, Any],
            user_id: Optional[str] = None,
            priority: int = 1
    ) -> SimulationResult:
        """Run a simulation with given parameters."""
        simulation_id = str(uuid.uuid4())[:8]

        try:
            # Validate parameters
            from core.utils.validators import validate_simulation_params
            is_valid, error_msg, validated_params = validate_simulation_params(parameters)

            if not is_valid:
                raise ValueError(f"Invalid parameters: {error_msg}")

            # Check cache
            cache_key = f"simulation_{hash(str(validated_params))}"
            cached_result = self.cache_service.get(cache_key)

            if cached_result:
                logger.info(f"Cache hit for simulation {simulation_id}")
                return cached_result

            # Register simulation
            self._register_simulation(simulation_id, validated_params, user_id)

            # Run simulation
            logger.info(f"Starting simulation {simulation_id}")
            result = await self.orchestrator.run_simulation(
                strategy=validated_params.get('strategy', 'security'),
                parameters=validated_params,
                priority=priority
            )

            # Cache result
            self.cache_service.set(cache_key, result, ttl=settings.CACHE_TTL)

            # Update simulation record
            self._update_simulation(simulation_id, 'completed', result)

            # Add to history
            self.simulation_history.append({
                'id': simulation_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': validated_params,
                'result': result.to_dict(),
                'user_id': user_id
            })

            logger.info(f"Simulation {simulation_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            self._update_simulation(simulation_id, 'failed', error=str(e))
            raise

    async def run_batch_simulations(
            self,
            simulations: List[Dict[str, Any]],
            user_id: Optional[str] = None
    ) -> List[SimulationResult]:
        """Run multiple simulations in batch."""
        tasks = []

        for params in simulations:
            task = self.run_simulation(params, user_id)
            tasks.append(task)

        # Run with limited concurrency
        semaphore = asyncio.Semaphore(self.orchestrator.max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        batch_tasks = [run_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, SimulationResult):
                valid_results.append(result)
            else:
                logger.error(f"Batch simulation failed: {result}")

        return valid_results

    def get_simulation_status(self, simulation_id: str) -> Optional[Dict]:
        """Get status of a simulation."""
        if simulation_id in self.active_simulations:
            return self.active_simulations[simulation_id]

        # Check history
        for record in self.simulation_history:
            if record['id'] == simulation_id:
                return {
                    'id': simulation_id,
                    'status': 'completed',
                    'timestamp': record['timestamp'],
                    'result': record['result']
                }

        return None

    def get_user_simulations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get simulations for a specific user."""
        user_simulations = [
            record for record in self.simulation_history
            if record.get('user_id') == user_id
        ]

        return sorted(
            user_simulations,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation service statistics."""
        total = len(self.simulation_history)
        completed = sum(1 for r in self.simulation_history if 'result' in r)
        failed = total - completed

        # Calculate average metrics
        if completed > 0:
            safety_scores = [
                r['result']['safety_score']
                for r in self.simulation_history
                if 'result' in r
            ]
            liberty_scores = [
                r['result']['liberty_score']
                for r in self.simulation_history
                if 'result' in r
            ]

            avg_safety = sum(safety_scores) / len(safety_scores)
            avg_liberty = sum(liberty_scores) / len(liberty_scores)
        else:
            avg_safety = avg_liberty = 0

        return {
            'total_simulations': total,
            'completed': completed,
            'failed': failed,
            'success_rate': (completed / total * 100) if total > 0 else 0,
            'average_safety': avg_safety,
            'average_liberty': avg_liberty,
            'active_simulations': len(self.active_simulations),
            'cache_hits': self.cache_service.get_stats().get('hits', 0)
        }

    def clear_history(self, older_than_days: Optional[int] = None):
        """Clear simulation history."""
        if older_than_days:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            self.simulation_history = [
                r for r in self.simulation_history
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
        else:
            self.simulation_history.clear()

        logger.info("Cleared simulation history")

    def _register_simulation(self, simulation_id: str, parameters: Dict, user_id: Optional[str]):
        """Register a new simulation."""
        self.active_simulations[simulation_id] = {
            'id': simulation_id,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'parameters': parameters,
            'user_id': user_id,
            'progress': 0.0
        }

    def _update_simulation(self, simulation_id: str, status: str, result: Any = None, error: Optional[str] = None):
        """Update simulation status."""
        if simulation_id in self.active_simulations:
            record = self.active_simulations[simulation_id]
            record['status'] = status
            record['end_time'] = datetime.now().isoformat()

            if result:
                record['result'] = result.to_dict() if hasattr(result, 'to_dict') else result

            if error:
                record['error'] = error

            # Move to history if completed
            if status in ['completed', 'failed', 'cancelled']:
                self.simulation_history.append(record.copy())
                del self.active_simulations[simulation_id]